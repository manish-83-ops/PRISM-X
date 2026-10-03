"""PRISMX Cross-Encoder Reranker using MiniLM-L6 INT8 with Deadline Governor and Adaptive Latency Optimization."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("prismx.rerank")


class CrossEncoderReranker:
    """MiniLM-L6 Cross-Encoder with dynamic INT8 quantization, length truncation, and deadline governor."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        torch_threads: int = 8,
    ) -> None:
        self.model_name = model_name
        self.torch_threads = torch_threads
        torch.set_num_threads(torch_threads)

        logger.info(f"Loading cross-encoder reranker {model_name} (threads={torch_threads})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # PyTorch Dynamic INT8 Quantization on Linear layers for ~2x speedup on CPU
        logger.info("Applying dynamic INT8 quantization to cross-encoder linear layers...")
        self.model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.model.eval()
        logger.info("Cross-encoder INT8 reranker successfully initialized.")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        max_length: int = 128,
        deadline_ms: float | None = 200.0,
        t_request_start: float | None = None,
        batch_size: int = 5,
    ) -> tuple[list[dict[str, Any]], float, str]:
        """Rerank candidate passages with deadline governor protection.

        Args:
            query: The search query string.
            candidates: List of candidate dicts, ordered by first-stage fused score.
            top_k: Number of reranked candidates to return.
            max_length: Sequence length truncation (default 128 for 2.6x speedup over 256).
            deadline_ms: Total request deadline in ms. If exceeded, returns best so far.
            t_request_start: Start timestamp of request (perf_counter).
            batch_size: Micro-batch size for iterative deadline checks.

        Returns:
            Tuple of (reranked_pool, rerank_latency_ms, governor_state).
        """
        if not candidates:
            return [], 0.0, "normal"

        t_rerank_start = time.perf_counter()
        governor_state = "normal"
        scored_candidates: list[dict[str, Any]] = []
        unscored_candidates: list[dict[str, Any]] = []

        total_candidates = len(candidates)

        for b_start in range(0, total_candidates, batch_size):
            # Check deadline before starting next micro-batch
            if deadline_ms is not None and t_request_start is not None:
                elapsed_total_ms = (time.perf_counter() - t_request_start) * 1000.0
                if elapsed_total_ms >= deadline_ms:
                    governor_state = "truncated" if scored_candidates else "exhausted_before_first_batch"
                    unscored_candidates.extend(candidates[b_start:])
                    break

            batch = candidates[b_start : b_start + batch_size]
            pairs = [[query, c.get("text", "")] for c in batch]

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
                    scores = [float(logits.item())]
                else:
                    scores = logits.tolist()

            for initial_rank_offset, (cand, score) in enumerate(zip(batch, scores)):
                initial_rank = b_start + initial_rank_offset + 1
                cand_copy = dict(cand)
                cand_copy["fused_rank"] = cand_copy.get("fused_rank") or initial_rank
                cand_copy["fused_score"] = cand_copy.get("fused_score") or cand.get("score")
                cand_copy["rerank_score"] = round(float(score), 4)
                scored_candidates.append(cand_copy)

        # Sort evaluated candidates by rerank_score descending
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        for new_rank, cand in enumerate(scored_candidates, start=1):
            cand["rerank_rank"] = new_rank

        # If governor truncated, append remaining unscored candidates in their original fused order
        if unscored_candidates:
            for unscored_rank, cand in enumerate(unscored_candidates, start=len(scored_candidates) + 1):
                cand_copy = dict(cand)
                cand_copy["fused_rank"] = cand_copy.get("fused_rank") or unscored_rank
                cand_copy["fused_score"] = cand_copy.get("fused_score") or cand.get("score")
                cand_copy["rerank_score"] = None
                cand_copy["rerank_rank"] = unscored_rank
                scored_candidates.append(cand_copy)

        dt_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)
        return scored_candidates[:top_k], dt_ms, governor_state
