"""PRISMX Hybrid Fusion Engine (Weighted Normalization and RRF) with Stable Tie-Breaking."""

from __future__ import annotations

import math
from typing import Any, Literal
import numpy as np

NormalizationType = Literal["minmax", "minmax_clipped", "zscore"]

def normalize_scores(
    scores: list[float],
    method: NormalizationType = "minmax",
) -> list[float]:
    """Normalizes candidate scores within a channel according to PATCH-3 specifications."""
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]

    arr = np.array(scores, dtype=np.float64)
    min_val, max_val = float(np.min(arr)), float(np.max(arr))

    # Zero-variance edge case
    if math.isclose(min_val, max_val):
        return [1.0] * len(scores)

    if method == "minmax":
        norm = (arr - min_val) / (max_val - min_val)
        return [float(x) for x in norm]

    elif method == "minmax_clipped":
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        if math.isclose(p5, p95):
            return [1.0] * len(scores)
        clipped = np.clip(arr, p5, p95)
        norm = (clipped - p5) / (p95 - p5)
        return [float(x) for x in norm]

    elif method == "zscore":
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        if math.isclose(std_val, 0.0):
            return [1.0] * len(scores)
        z = (arr - mean_val) / std_val
        # Sigmoid squash to [0, 1] range for intuitive weighted combination
        norm = 1.0 / (1.0 + np.exp(-z))
        return [float(x) for x in norm]

    else:
        raise ValueError(f"Unknown normalization method: {method}")

class FusionEngine:
    @staticmethod
    def fuse_weighted(
        dense_candidates: list[dict[str, Any]],
        sparse_candidates: list[dict[str, Any]],
        alpha: float = 0.7,
        norm_method: NormalizationType = "minmax",
    ) -> list[dict[str, Any]]:
        """Fuses dense and sparse candidate lists using weighted combination."""
        dense_map = {c["passage_id"]: c for c in dense_candidates}
        sparse_map = {c["passage_id"]: c for c in sparse_candidates}

        # Normalize dense channel
        dense_raw = [c["dense_score"] for c in dense_candidates]
        dense_norm = normalize_scores(dense_raw, method=norm_method)
        dense_norm_map = {c["passage_id"]: dense_norm[i] for i, c in enumerate(dense_candidates)}

        # Normalize sparse channel
        sparse_raw = [c["bm25_score"] for c in sparse_candidates]
        sparse_norm = normalize_scores(sparse_raw, method=norm_method)
        sparse_norm_map = {c["passage_id"]: sparse_norm[i] for i, c in enumerate(sparse_candidates)}

        # Union of all candidate passage IDs
        all_ids = sorted(set(dense_map.keys()).union(set(sparse_map.keys())))

        fused = []
        for pid in all_ids:
            s_dense_n = dense_norm_map.get(pid, 0.0)
            s_sparse_n = sparse_norm_map.get(pid, 0.0)
            fused_score = alpha * s_dense_n + (1.0 - alpha) * s_sparse_n

            d_entry = dense_map.get(pid, {})
            s_entry = sparse_map.get(pid, {})

            fused.append({
                "passage_id": pid,
                "score": float(fused_score),
                "dense_rank": d_entry.get("dense_rank"),
                "dense_score": d_entry.get("dense_score"),
                "bm25_rank": s_entry.get("bm25_rank"),
                "bm25_score": s_entry.get("bm25_score"),
                "category": d_entry.get("category") or s_entry.get("category"),
                "source": d_entry.get("source") or s_entry.get("source"),
            })

        # Stable tie-breaking: fused_score DESC, dense_rank ASC, bm25_rank ASC, passage_id ASC
        def sort_key(item: dict[str, Any]):
            d_rank = item["dense_rank"] if item["dense_rank"] is not None else float("inf")
            s_rank = item["bm25_rank"] if item["bm25_rank"] is not None else float("inf")
            return (-item["score"], d_rank, s_rank, str(item["passage_id"]))

        fused.sort(key=sort_key)
        for rank, item in enumerate(fused, start=1):
            item["rank"] = rank

        return fused

    @staticmethod
    def fuse_rrf(
        dense_candidates: list[dict[str, Any]],
        sparse_candidates: list[dict[str, Any]],
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Fuses dense and sparse candidate lists using Reciprocal Rank Fusion (RRF)."""
        dense_map = {c["passage_id"]: c for c in dense_candidates}
        sparse_map = {c["passage_id"]: c for c in sparse_candidates}

        all_ids = sorted(set(dense_map.keys()).union(set(sparse_map.keys())))

        fused = []
        for pid in all_ids:
            score = 0.0
            d_entry = dense_map.get(pid, {})
            s_entry = sparse_map.get(pid, {})

            if "dense_rank" in d_entry and d_entry["dense_rank"] is not None:
                score += 1.0 / (rrf_k + d_entry["dense_rank"])
            if "bm25_rank" in s_entry and s_entry["bm25_rank"] is not None:
                score += 1.0 / (rrf_k + s_entry["bm25_rank"])

            fused.append({
                "passage_id": pid,
                "score": float(score),
                "dense_rank": d_entry.get("dense_rank"),
                "dense_score": d_entry.get("dense_score"),
                "bm25_rank": s_entry.get("bm25_rank"),
                "bm25_score": s_entry.get("bm25_score"),
                "category": d_entry.get("category") or s_entry.get("category"),
                "source": d_entry.get("source") or s_entry.get("source"),
            })

        def sort_key(item: dict[str, Any]):
            d_rank = item["dense_rank"] if item["dense_rank"] is not None else float("inf")
            s_rank = item["bm25_rank"] if item["bm25_rank"] is not None else float("inf")
            return (-item["score"], d_rank, s_rank, str(item["passage_id"]))

        fused.sort(key=sort_key)
        for rank, item in enumerate(fused, start=1):
            item["rank"] = rank

        return fused
