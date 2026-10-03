import sys
from pathlib import Path
import time
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prismx.retrieve.rerank import CrossEncoderReranker


def test_reranker_initialization_and_calibration():
    reranker = CrossEncoderReranker()
    assert reranker.session is not None or hasattr(reranker, "model")
    assert reranker.rolling_per_pair_ms > 0.0
    print(f"Calibrated per_pair_ms: {reranker.rolling_per_pair_ms:.2f} ms")


def test_reranker_normal_execution():
    reranker = CrossEncoderReranker()
    candidates = [
        {"passage_id": f"p_{i}", "text": f"This is candidate passage number {i} discussing artificial intelligence.", "score": 0.8 - i * 0.05}
        for i in range(10)
    ]
    t0 = time.perf_counter()
    results, dt_ms, gov_state, scored_count, per_pair_ms = reranker.rerank(
        query="what is artificial intelligence",
        candidates=candidates,
        top_k=5,
        total_deadline_ms=500.0,
        t_request_start=t0,
        batch_size=2,
    )
    assert len(results) == 5
    assert gov_state == "normal"
    assert scored_count == 5
    assert all(r["rerank_score"] is not None for r in results)
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]
    assert results[0]["rerank_rank"] == 1


def test_reranker_skipped_budget():
    reranker = CrossEncoderReranker()
    candidates = [
        {"passage_id": f"p_{i}", "text": f"Candidate {i}", "score": 0.9 - i * 0.1}
        for i in range(10)
    ]
    # Request start simulated 300ms ago -> deadline 230ms already exhausted!
    t0 = time.perf_counter() - 0.300
    results, dt_ms, gov_state, scored_count, per_pair_ms = reranker.rerank(
        query="test query",
        candidates=candidates,
        top_k=5,
        total_deadline_ms=230.0,
        t_request_start=t0,
        batch_size=2,
    )
    assert len(results) == 5
    assert gov_state == "skipped_budget"
    assert scored_count == 0
    # Original order must be preserved!
    assert [r["passage_id"] for r in results] == [f"p_{i}" for i in range(5)]
    assert all(r["rerank_score"] is None for r in results)


def test_reranker_truncated_governor():
    reranker = CrossEncoderReranker()
    candidates = [
        {"passage_id": f"p_{i}", "text": f"Candidate passage {i} about renewable solar energy systems.", "score": 0.9 - i * 0.05}
        for i in range(10)
    ]
    # Simulate a request where remaining time only allows scoring ~2 candidates
    per_pair = reranker.rolling_per_pair_ms
    # Set total deadline so remaining budget is exactly ~2.5 * per_pair + reserve
    reserve = 4.0
    budget = (2.2 * per_pair) + reserve
    t0 = time.perf_counter()
    results, dt_ms, gov_state, scored_count, per_pair_ms = reranker.rerank(
        query="renewable solar energy",
        candidates=candidates,
        top_k=5,
        total_deadline_ms=budget,
        t_request_start=t0,
        reserve_ms=reserve,
        batch_size=2,
    )
    assert len(results) == 5
    assert gov_state == "truncated"
    assert scored_count == 2
    # Scored candidates are first 2, sorted by rerank_score
    assert results[0]["rerank_score"] is not None
    assert results[1]["rerank_score"] is not None
    # Unscored candidates are below them in first-stage order
    assert results[2]["rerank_score"] is None
    assert results[2]["passage_id"] == "p_2"
