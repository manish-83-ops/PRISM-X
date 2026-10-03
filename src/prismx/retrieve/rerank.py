"""PRISMX Cross-Encoder Reranker using ONNX Runtime FP32 with Anytime Cascade and Predictive Deadline Governor.
Conforms to ADR-022:
- ONNX Runtime FP32 session with intra_op = physical cores, inter_op = 1, allow_spinning = 0, graph optimizations on.
- Zero torch in serving path (numpy tensors with transformers tokenizer).
- Startup calibration (20 warm pairs) to seed rolling median per-pair cost.
- Request-level deadline: t0 = request arrival (ASGI middleware).
  remaining = total_deadline_ms(230) - elapsed - reserve (p95 of fetch+serialize = ~4.0 ms).
- K_eff = clamp(floor(remaining / per_pair_ms), 0, K).
- Micro-batches (2-3 pairs) with hard deadline check between batches.
- Candidates beyond scored keep first-stage order below scored.
- States: normal / truncated / skipped_budget.
"""

from __future__ import annotations

import collections
import logging
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import psutil
from transformers import AutoTokenizer

logger = logging.getLogger("prismx.rerank")

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ONNX_MODEL_PATH = Path("models/onnx_cross_encoder/model_fp32.onnx")


class CrossEncoderReranker:
    """MiniLM-L6 Cross-Encoder ONNX Runtime FP32 with Anytime Cascade and Predictive Governor."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        onnx_path: str | Path | None = None,
        intra_op_threads: int | None = None,
        torch_threads: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.onnx_path = Path(onnx_path) if onnx_path else ONNX_MODEL_PATH
        self.use_onnx = False
        self.session = None
        self._pair_latencies: collections.deque[float] = collections.deque(maxlen=100)

        # 1. Initialize Tokenizer
        logger.info(f"Loading cross-encoder tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 2. Initialize Inference Session (ONNX Runtime preferred)
        physical_cores = psutil.cpu_count(logical=False) or 6
        self.intra_op_threads = intra_op_threads or torch_threads or physical_cores

        if self.onnx_path.exists():
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self.intra_op_threads
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

                logger.info(
                    f"Initializing ONNX Runtime FP32 session from {self.onnx_path} "
                    f"(intra_op={self.intra_op_threads}, inter_op=1, spinning=0)..."
                )
                self.session = ort.InferenceSession(str(self.onnx_path), opts)
                self.use_onnx = True
                logger.info("ONNX Runtime cross-encoder initialized successfully (serving path torch-free).")
            except Exception as exc:
                logger.warning(f"Failed to load ONNX model: {exc}. Falling back to PyTorch.")
                self.use_onnx = False

        if not self.use_onnx:
            # Fallback to PyTorch Dynamic INT8
            import torch
            from transformers import AutoModelForSequenceClassification

            logger.info("Initializing PyTorch fallback model with dynamic INT8...")
            torch.set_num_threads(self.intra_op_threads)
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model = torch.quantization.quantize_dynamic(
                base_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.model.eval()

        # 3. Startup Calibration: Run 20 warm pairs to seed rolling median per-pair latency
        self._calibrate_startup(warm_pairs=20)

    def _calibrate_startup(self, warm_pairs: int = 20) -> None:
        """Runs startup warm-up pairs to measure CPU forward latency and initialize rolling median."""
        dummy_query = "what is the capital of france"
        dummy_texts = [
            f"Paris is the capital and most populous city of France with official population {i}."
            for i in range(warm_pairs)
        ]
        pairs = [[dummy_query, text] for text in dummy_texts]

        try:
            t0 = time.perf_counter()
            self._forward_pairs(pairs[:2], max_length=128)  # prime session
            for i in range(0, warm_pairs, 2):
                chunk = pairs[i : i + 2]
                t_chunk = time.perf_counter()
                self._forward_pairs(chunk, max_length=128)
                chunk_ms = (time.perf_counter() - t_chunk) * 1000.0
                per_pair = chunk_ms / len(chunk)
                self._pair_latencies.append(per_pair)

            total_ms = (time.perf_counter() - t0) * 1000.0
            med_ms = self.rolling_per_pair_ms
            logger.info(
                f"Cross-encoder calibration complete: {warm_pairs} warm pairs in {total_ms:.1f}ms. "
                f"Rolling median per-pair latency: {med_ms:.2f}ms"
            )
        except Exception as exc:
            logger.warning(f"Startup calibration encountered an error: {exc}. Using default 7.5ms/pair.")
            self._pair_latencies.extend([7.5] * 10)

    @property
    def rolling_per_pair_ms(self) -> float:
        """Returns the rolling median per-pair latency estimate."""
        if not self._pair_latencies:
            return 7.5
        return float(statistics.median(self._pair_latencies))

    def _forward_pairs(self, pairs: list[list[str]], max_length: int = 128) -> list[float]:
        """Runs forward inference on pairs. Uses ONNX Runtime (numpy) if active, else PyTorch."""
        if not pairs:
            return []

        if self.use_onnx and self.session is not None:
            # Tokenize returning numpy arrays
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            # Length-sorted dynamic padding: inputs already padded dynamically to max length in pairs
            ort_inputs = {k: v for k, v in inputs.items()}
            outputs = self.session.run(None, ort_inputs)
            logits = outputs[0].squeeze(-1)
            if logits.ndim == 0:
                return [float(logits.item())]
            return [float(x) for x in logits]
        else:
            import torch

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)
                if logits.dim() == 0:
                    return [float(logits.item())]
                return [float(x) for x in logits.tolist()]

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        max_length: int = 128,
        total_deadline_ms: float = 230.0,
        t_request_start: float | None = None,
        reserve_ms: float = 4.0,
        batch_size: int = 2,
    ) -> tuple[list[dict[str, Any]], float, str, int, float]:
        """Reranks candidate passages using predictive anytime cascade semantics.

        Args:
            query: The search query string.
            candidates: List of candidate dicts in first-stage order.
            top_k: Number of candidates to score and return.
            max_length: Sequence truncation length (default 128).
            total_deadline_ms: Request-level SLA deadline (default 230ms).
            t_request_start: Request arrival timestamp (perf_counter) from ASGI middleware.
            reserve_ms: Measured p95 of fetch + serialization safety buffer (default 4.0ms).
            batch_size: Micro-batch size (default 2 pairs) for iterative deadline checks.

        Returns:
            Tuple of:
            (reranked_candidates, latency_ms, governor_state, candidates_scored, per_pair_ms)
        """
        if not candidates:
            return [], 0.0, "normal", 0, self.rolling_per_pair_ms

        t_rerank_start = time.perf_counter()
        t0 = t_request_start if t_request_start is not None else t_rerank_start
        per_pair_est = self.rolling_per_pair_ms

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        remaining_ms = total_deadline_ms - elapsed_ms - reserve_ms

        K_requested = min(top_k, len(candidates))

        # Predictive K_eff calculation:
        # K_eff = clamp(floor(remaining / per_pair_ms), 0, K_requested)
        if remaining_ms <= 0 or per_pair_est <= 0:
            K_eff = 0
        else:
            K_eff = max(0, min(K_requested, math.floor(remaining_ms / per_pair_est)))

        # If zero budget remaining before reranking starts:
        if K_eff == 0:
            logger.info(
                f"Governor skipped reranking: remaining={remaining_ms:.1f}ms <= 0 "
                f"(per_pair={per_pair_est:.2f}ms). Falling back to first-stage order."
            )
            # Candidates keep first-stage order
            unaltered: list[dict[str, Any]] = []
            for rank_idx, cand in enumerate(candidates, start=1):
                c = dict(cand)
                c["fused_rank"] = c.get("fused_rank") or rank_idx
                c["fused_score"] = c.get("fused_score") or c.get("score")
                c["rerank_score"] = None
                c["rerank_rank"] = rank_idx
                unaltered.append(c)
            dt_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)
            return unaltered[:top_k], dt_ms, "skipped_budget", 0, round(per_pair_est, 2)

        # Micro-batch evaluation of top-K_eff candidates
        eval_pool = candidates[:K_eff]
        scored_candidates: list[dict[str, Any]] = []
        stopped_early = False

        for b_start in range(0, len(eval_pool), batch_size):
            # Hard deadline check before starting next micro-batch
            current_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            current_remaining_ms = total_deadline_ms - current_elapsed_ms - reserve_ms
            if current_remaining_ms < per_pair_est:
                stopped_early = True
                logger.info(
                    f"Governor truncated between micro-batches: remaining={current_remaining_ms:.1f}ms < "
                    f"per_pair={per_pair_est:.2f}ms. Evaluated {len(scored_candidates)} of {K_requested}."
                )
                break

            batch = eval_pool[b_start : b_start + batch_size]
            pairs = [[query, c.get("text", "")] for c in batch]

            t_batch_start = time.perf_counter()
            scores = self._forward_pairs(pairs, max_length=max_length)
            batch_ms = (time.perf_counter() - t_batch_start) * 1000.0

            # Record observed per-pair latency to update rolling median
            if len(batch) > 0:
                observed_per_pair = batch_ms / len(batch)
                self._pair_latencies.append(observed_per_pair)

            for rank_offset, (cand, score) in enumerate(zip(batch, scores)):
                c = dict(cand)
                c["fused_rank"] = c.get("fused_rank") or (b_start + rank_offset + 1)
                c["fused_score"] = c.get("fused_score") or cand.get("score")
                c["rerank_score"] = round(float(score), 4)
                scored_candidates.append(c)

        candidates_scored = len(scored_candidates)

        # Determine governor state
        if candidates_scored == 0:
            governor_state = "skipped_budget"
        elif candidates_scored < K_requested:
            governor_state = "truncated"
        else:
            governor_state = "normal"

        # Sort scored candidates by rerank_score descending
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        for rank_idx, cand in enumerate(scored_candidates, start=1):
            cand["rerank_rank"] = rank_idx

        # Candidates beyond scored keep first-stage order below the scored ones
        unscored_candidates = candidates[candidates_scored:]
        final_pool = list(scored_candidates)
        for offset, cand in enumerate(unscored_candidates, start=candidates_scored + 1):
            c = dict(cand)
            c["fused_rank"] = c.get("fused_rank") or offset
            c["fused_score"] = c.get("fused_score") or cand.get("score")
            c["rerank_score"] = None
            c["rerank_rank"] = offset
            final_pool.append(c)

        dt_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)
        return final_pool[:top_k], dt_ms, governor_state, candidates_scored, round(self.rolling_per_pair_ms, 2)
