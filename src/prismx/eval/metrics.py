"""PRISMX Information Retrieval Metrics and Secondary Duplicate-Aware Evaluators."""

from __future__ import annotations

import math
from typing import Sequence

def hit_at_1(retrieved_ids: Sequence[str], gold_ids: set[str]) -> float:
    """Success@1: 1.0 if top-1 passage is relevant, else 0.0."""
    if not retrieved_ids or not gold_ids:
        return 0.0
    return 1.0 if retrieved_ids[0] in gold_ids else 0.0

def success_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int = 5) -> float:
    """Success@k: 1.0 if at least one relevant passage appears in top-k, else 0.0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if len(top_k.intersection(gold_ids)) > 0 else 0.0

def recall_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int = 20) -> float:
    """Recall@k: |gold ∩ top_k| / |gold|."""
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(gold_ids))
    return hits / len(gold_ids)

def mrr_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int = 10) -> float:
    """Mean Reciprocal Rank @ k: 1 / rank of first relevant passage in top-k, else 0.0."""
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if pid in gold_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @ k with binary relevance gains."""
    if not gold_ids or not retrieved_ids:
        return 0.0

    dcg = 0.0
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if pid in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: min(|gold|, k) relevant items placed at ranks 1..ideal
    num_ideal = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, num_ideal + 1))

    return dcg / idcg if idcg > 0.0 else 0.0

def dup_aware_hit_at_1(
    retrieved_ids: Sequence[str],
    gold_ids: set[str],
    near_dups_map: dict[str, list[dict]] | None = None,
) -> float:
    """Secondary metric (PATCH-4): Near-duplicates of gold passage also count as gold."""
    if not retrieved_ids:
        return 0.0
    top_1 = retrieved_ids[0]
    if top_1 in gold_ids:
        return 1.0
    if near_dups_map:
        for gid in gold_ids:
            dups = {d["passage_id"] for d in near_dups_map.get(gid, [])}
            if top_1 in dups:
                return 1.0
    return 0.0

def dup_aware_recall_at_5(
    retrieved_ids: Sequence[str],
    gold_ids: set[str],
    near_dups_map: dict[str, list[dict]] | None = None,
) -> float:
    """Secondary metric (PATCH-4): Near-duplicates of gold passage also count as gold."""
    if not gold_ids:
        return 0.0
    top_5 = set(retrieved_ids[:5])
    expanded_gold = set(gold_ids)
    if near_dups_map:
        for gid in gold_ids:
            for d in near_dups_map.get(gid, []):
                expanded_gold.add(d["passage_id"])

    hits = len(top_5.intersection(expanded_gold))
    return min(1.0, hits / len(gold_ids))
