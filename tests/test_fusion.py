"""Unit tests for Hybrid Fusion Engine (PATCH-3, RRF, Weighted combination)."""

import math
import pytest
from prismx.retrieve.fusion import FusionEngine, normalize_scores


def test_normalize_scores_edge_cases():
    # Empty
    assert normalize_scores([]) == []
    # Single score
    assert normalize_scores([5.0]) == [1.0]
    # Zero-variance (all identical)
    res = normalize_scores([3.0, 3.0, 3.0], method="minmax")
    assert res == [1.0, 1.0, 1.0]

    res_clip = normalize_scores([3.0, 3.0, 3.0], method="minmax_clipped")
    assert res_clip == [1.0, 1.0, 1.0]

    res_z = normalize_scores([3.0, 3.0, 3.0], method="zscore")
    assert res_z == [1.0, 1.0, 1.0]


def test_normalize_scores_minmax():
    scores = [10.0, 20.0, 30.0]
    norm = normalize_scores(scores, method="minmax")
    assert math.isclose(norm[0], 0.0)
    assert math.isclose(norm[1], 0.5)
    assert math.isclose(norm[2], 1.0)


def test_normalize_scores_minmax_clipped():
    # Extreme outlier at the high end
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]
    norm = normalize_scores(scores, method="minmax_clipped")
    assert len(norm) == len(scores)
    # The outlier (1000.0) should be clipped to the 95th percentile, giving norm 1.0
    assert math.isclose(norm[-1], 1.0)
    # The min should be 0.0
    assert math.isclose(norm[0], 0.0)


def test_normalize_scores_zscore():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    norm = normalize_scores(scores, method="zscore")
    # Mean is 3.0, so score 3.0 should have z = 0, sigmoid(0) = 0.5
    assert math.isclose(norm[2], 0.5, abs_tol=1e-5)
    # Below mean should be < 0.5, above mean > 0.5
    assert norm[0] < 0.5
    assert norm[-1] > 0.5


def test_fuse_weighted_basic():
    dense_candidates = [
        {"passage_id": "doc1", "dense_score": 0.9, "dense_rank": 1},
        {"passage_id": "doc2", "dense_score": 0.7, "dense_rank": 2},
    ]
    sparse_candidates = [
        {"passage_id": "doc2", "bm25_score": 15.0, "bm25_rank": 1},
        {"passage_id": "doc3", "bm25_score": 10.0, "bm25_rank": 2},
    ]

    # alpha = 0.5
    fused = FusionEngine.fuse_weighted(dense_candidates, sparse_candidates, alpha=0.5, norm_method="minmax")
    assert len(fused) == 3

    # doc2 appears in both channels:
    # dense_norm: (0.7 - 0.7)/(0.9 - 0.7) = 0.0
    # sparse_norm: (15.0 - 10.0)/(15.0 - 10.0) = 1.0
    # score = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    doc2_entry = next(c for c in fused if c["passage_id"] == "doc2")
    assert math.isclose(doc2_entry["score"], 0.5)
    assert doc2_entry["dense_rank"] == 2
    assert doc2_entry["bm25_rank"] == 1


def test_fuse_rrf():
    dense_candidates = [
        {"passage_id": "doc1", "dense_score": 0.9, "dense_rank": 1},
        {"passage_id": "doc2", "dense_score": 0.8, "dense_rank": 2},
    ]
    sparse_candidates = [
        {"passage_id": "doc2", "bm25_score": 12.0, "bm25_rank": 1},
        {"passage_id": "doc3", "bm25_score": 10.0, "bm25_rank": 2},
    ]

    fused = FusionEngine.fuse_rrf(dense_candidates, sparse_candidates, rrf_k=60)
    assert len(fused) == 3

    # doc2 appears in both: 1/(60 + 2) + 1/(60 + 1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    expected_doc2_score = (1.0 / 62.0) + (1.0 / 61.0)
    doc2_entry = next(c for c in fused if c["passage_id"] == "doc2")
    assert math.isclose(doc2_entry["score"], expected_doc2_score, rel_tol=1e-4)

    # doc2 should be ranked 1 because of dual-channel presence
    assert fused[0]["passage_id"] == "doc2"
    assert fused[0]["rank"] == 1


def test_stable_tie_breaking():
    # Two passages with identical scores
    dense_cands = [
        {"passage_id": "doc_b", "dense_score": 0.5, "dense_rank": 1},
        {"passage_id": "doc_a", "dense_score": 0.5, "dense_rank": 2},
    ]
    sparse_cands = []

    fused = FusionEngine.fuse_weighted(dense_cands, sparse_cands, alpha=1.0)
    # Both have normalized score 1.0 (zero variance in input scores)
    # Tie-breaking rule: fused_score DESC, dense_rank ASC, bm25_rank ASC, passage_id ASC
    # doc_b has dense_rank 1, doc_a has dense_rank 2 -> doc_b must come first
    assert fused[0]["passage_id"] == "doc_b"
    assert fused[1]["passage_id"] == "doc_a"
