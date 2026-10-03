"""PRISMX Cross-Encoder Reranker using MiniLM-L6 with PyTorch Dynamic INT8 Quantization."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("prismx.rerank")


class CrossEncoderReranker:
    """MiniLM-L6 Cross-Encoder with dynamic INT8 quantization for sub-50ms CPU reranking."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        torch_threads: int = 12,
    ) -> None:
        self.model_name = model_name
        self.torch_threads = torch_threads
        torch.set_num_threads(torch_threads)

        logger.info(f"Loading cross-encoder reranker {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # PyTorch Dynamic INT8 Quantization on Linear layers for 2-3x speedup on CPU
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
    ) -> tuple[list[dict[str, Any]], float]:
        """Rerank candidate passages against the query using cross-attention.

        Args:
            query: The search query string.
            candidates: List of candidate dicts, each must contain 'text' and 'passage_id'.
            top_k: Number of reranked candidates to return.

        Returns:
            Tuple of (reranked_candidates_list, rerank_latency_ms).
        """
        if not candidates:
            return [], 0.0

        t0 = time.perf_counter()
        pairs = [[query, c.get("text", "")] for c in candidates]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1)
            if logits.dim() == 0:
                scores = [float(logits.item())]
            else:
                scores = logits.tolist()

        scored_candidates: list[dict[str, Any]] = []
        for initial_rank, (cand, score) in enumerate(zip(candidates, scores), start=1):
            cand_copy = dict(cand)
            cand_copy["fused_rank"] = cand_copy.get("fused_rank") or initial_rank
            cand_copy["fused_score"] = cand_copy.get("fused_score") or cand.get("score")
            cand_copy["rerank_score"] = round(float(score), 4)
            scored_candidates.append(cand_copy)

        # Sort descending by rerank_score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        for new_rank, cand in enumerate(scored_candidates, start=1):
            cand["rerank_rank"] = new_rank

        dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return scored_candidates[:top_k], dt_ms
