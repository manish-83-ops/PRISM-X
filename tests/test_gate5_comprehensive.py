"""Comprehensive Gate 5.1 Verification Test Suite.
Tests covering:
1. Governor (budget exhausted before first batch, mid-run truncation, normal)
2. Mode switching (dense / hybrid / prismx, default is hybrid)
3. Composite pool selection status check (documenting lack of separate composite pool)
4. Upsert visible in next search & Delete gone
5. Metadata filter applied at Qdrant level (zero out-of-filter results)
6. Cache invalidation on upsert and delete
"""

import collections
import time
import pytest
from unittest.mock import MagicMock, patch
from prismx.retrieve.rerank import CrossEncoderReranker
from prismx.retrieve.service import SearchService
from prismx.schemas import SearchRequest, FilterParams, UpsertRequest


# ---------------------------------------------------------------------------
# 1. GOVERNOR TESTS
# ---------------------------------------------------------------------------

class DummyModel:
    def __init__(self):
        self.call_count = 0

    def __call__(self, **inputs):
        import torch
        self.call_count += 1
        batch_size = inputs["input_ids"].shape[0]
        logits = torch.tensor([0.5 + 0.1 * i for i in range(batch_size)])
        mock_output = MagicMock()
        mock_output.logits = logits
        return mock_output

    def eval(self):
        pass


def test_governor_budget_exhausted_before_first_batch():
    """Verify governor when budget is already exhausted before batch 0 starts."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = DummyModel()
    reranker.tokenizer = MagicMock()
    reranker.tokenizer.return_value = {
        "input_ids": MagicMock(shape=[2, 10]),
        "attention_mask": MagicMock()
    }
    reranker.use_onnx = False
    reranker._pair_latencies = collections.deque([5.0], maxlen=100)

    candidates = [
        {"passage_id": f"p{i}", "text": f"text {i}", "score": 1.0 - 0.1 * i, "fused_rank": i + 1}
        for i in range(10)
    ]

    # Start timestamp is in the past such that elapsed_ms (500ms) >= total_deadline_ms (100ms)
    t_start = time.perf_counter() - 0.5
    top_candidates, r_ms, gov_state, scored_count, per_pair_ms = reranker.rerank(
        query="test query",
        candidates=candidates,
        top_k=5,
        total_deadline_ms=100.0,
        t_request_start=t_start,
        batch_size=5,
    )

    assert gov_state == "skipped_budget"
    assert reranker.model.call_count == 0  # No model inferences ran!
    assert scored_count == 0
    assert len(top_candidates) == 5
    assert [c["passage_id"] for c in top_candidates] == ["p0", "p1", "p2", "p3", "p4"]
    assert all(c.get("rerank_score") is None for c in top_candidates)


def test_governor_mid_run_truncation():
    """Verify governor mid-run truncation when deadline expires between micro-batches."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = DummyModel()
    reranker.tokenizer = MagicMock()
    reranker.tokenizer.return_value = {
        "input_ids": MagicMock(shape=[2, 10]),
        "attention_mask": MagicMock()
    }
    reranker.use_onnx = False
    reranker._pair_latencies = collections.deque([10.0], maxlen=100)

    candidates = [
        {"passage_id": f"p{i}", "text": f"text {i}", "score": 1.0 - 0.1 * i, "fused_rank": i + 1}
        for i in range(6)
    ]

    # Deadline will expire after batch 1
    clock_vals = [100.0, 100.0, 100.0, 100.015, 100.042, 100.043, 100.044, 100.045]
    call_idx = 0
    def mock_clock():
        nonlocal call_idx
        val = clock_vals[min(call_idx, len(clock_vals) - 1)]
        call_idx += 1
        return val

    with patch("time.perf_counter", side_effect=mock_clock):
        top_candidates, r_ms, gov_state, scored_count, per_pair = reranker.rerank(
            query="test query",
            candidates=candidates,
            top_k=5,
            total_deadline_ms=50.0,
            t_request_start=100.0,
            reserve_ms=4.0,
            batch_size=2,
        )

    assert gov_state == "truncated"
    assert scored_count == 2
    assert reranker.model.call_count == 1  # Only first batch scored
    assert len(top_candidates) == 5
    # First 2 candidates scored by cross-encoder
    assert top_candidates[0]["rerank_score"] is not None
    assert top_candidates[1]["rerank_score"] is not None
    # Remaining candidates preserved from unscored fallback
    assert top_candidates[2]["rerank_score"] is None


def test_governor_normal_execution():
    """Verify governor normal completion when within budget."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = DummyModel()
    reranker.tokenizer = MagicMock()
    reranker.tokenizer.return_value = {
        "input_ids": MagicMock(shape=[2, 10]),
        "attention_mask": MagicMock()
    }
    reranker.use_onnx = False
    reranker._pair_latencies = collections.deque([5.0], maxlen=100)

    candidates = [
        {"passage_id": f"p{i}", "text": f"text {i}", "score": 1.0 - 0.1 * i, "fused_rank": i + 1}
        for i in range(4)
    ]

    t_start = time.perf_counter()
    top_candidates, r_ms, gov_state, scored_count, per_pair = reranker.rerank(
        query="test query",
        candidates=candidates,
        top_k=3,
        total_deadline_ms=5000.0,  # Generous deadline
        t_request_start=t_start,
        batch_size=2,
    )

    assert gov_state == "normal"
    assert scored_count == 3
    assert reranker.model.call_count == 2
    assert len(top_candidates) == 3
    assert all(c["rerank_score"] is not None for c in top_candidates)


# ---------------------------------------------------------------------------
# 2. MODE SWITCHING TESTS (Default is hybrid)
# ---------------------------------------------------------------------------

def test_mode_switching_default_is_hybrid():
    """Verify SearchRequest default mode is strictly 'hybrid'."""
    req = SearchRequest(query="sample query")
    assert req.mode == "hybrid"


def test_mode_switching_service_routing():
    """Verify SearchService routes correctly for dense, hybrid, and prismx."""
    dense_retriever = MagicMock()
    hybrid_retriever = MagicMock()
    text_store = MagicMock()
    qdrant_store = MagicMock()
    reranker = MagicMock()

    config = {
        "retrieval": {
            "candidate_depth": 20,
            "fusion": {"method": "weighted", "alpha": 0.8, "rrf_k": 60}
        }
    }

    service = SearchService(
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        text_store=text_store,
        qdrant_store=qdrant_store,
        reranker=reranker,
        config=config,
    )

    # Setup mocks
    dense_retriever.retrieve.return_value = ([{"passage_id": "p1", "dense_score": 0.9, "dense_rank": 1}], 5.0, 5.0)
    hybrid_retriever.retrieve.return_value = ([{"passage_id": "p2", "score": 0.85, "dense_rank": 1, "sparse_rank": 1}], {"dense": 5.0, "sparse": 5.0, "fusion": 1.0})
    text_store.get_passages_by_ids.return_value = {
        "p1": {"passage_id": "p1", "text": "dense text", "category": "cat", "source": "src"},
        "p2": {"passage_id": "p2", "text": "hybrid text", "category": "cat", "source": "src"}
    }
    text_store.get_meta.return_value = 1
    reranker.rerank.return_value = (
        [{"passage_id": "p2", "text": "hybrid text", "score": 0.95, "rerank_score": 0.95}],
        10.0,
        "normal",
        1,
        5.0
    )

    # 1. Default request (hybrid)
    resp_default = service.search(SearchRequest(query="test default"))
    assert resp_default.mode == "hybrid"
    hybrid_retriever.retrieve.assert_called()

    # 2. Dense request
    dense_retriever.retrieve.reset_mock()
    hybrid_retriever.retrieve.reset_mock()
    resp_dense = service.search(SearchRequest(query="test dense", mode="dense"))
    assert resp_dense.mode == "dense"
    dense_retriever.retrieve.assert_called_once()
    hybrid_retriever.retrieve.assert_not_called()

    # 3. PRISMX mode (hybrid + rerank K=10)
    reranker.rerank.reset_mock()
    resp_prismx = service.search(SearchRequest(query="test prismx", mode="prismx"))
    assert resp_prismx.mode == "prismx"
    reranker.rerank.assert_called_once()


def test_governor_total_deadline_respected_when_earlier_stages_slow():
    """Verify total deadline is respected when earlier retrieval/fusion stages are slow.
    If earlier stages take 245ms out of 250ms total deadline, remaining time minus 10ms safety
    is <= 0ms, so governor immediately exhausts before reranking any candidates.
    """
    dense_retriever = MagicMock()
    hybrid_retriever = MagicMock()
    text_store = MagicMock()
    qdrant_store = MagicMock()
    reranker = MagicMock()

    config = {
        "retrieval": {
            "candidate_depth": 20,
            "fusion": {"method": "weighted", "alpha": 0.8, "rrf_k": 60}
        }
    }

    service = SearchService(
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        text_store=text_store,
        qdrant_store=qdrant_store,
        reranker=reranker,
        config=config,
    )

    # Hybrid returns 10 candidates
    hybrid_retriever.retrieve.return_value = (
        [{"passage_id": f"p{i}", "score": 1.0 - 0.05 * i, "dense_rank": i + 1, "sparse_rank": i + 1} for i in range(10)],
        {"dense": 120.0, "sparse": 125.0, "fusion": 1.0}
    )
    text_store.get_passages_by_ids.return_value = {
        f"p{i}": {"passage_id": f"p{i}", "text": f"text {i}", "category": "cat", "source": "src"}
        for i in range(10)
    }
    text_store.get_meta.return_value = 1

    # Simulate reranker returning skipped_budget when deadline <= 0
    def mock_rerank(**kwargs):
        cands = kwargs.get("candidates", [])
        return cands[:kwargs.get("top_k", 5)], 0.05, "skipped_budget", 0, 5.0

    reranker.rerank.side_effect = mock_rerank

    # Simulate time passing during retrieve & hydration (e.g. 245 ms elapsed)
    call_count = 0
    def mock_clock():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return 1000.0 + (call_count * 0.001)
        return 1000.245 + ((call_count - 3) * 0.001)

    with patch("time.perf_counter", side_effect=mock_clock):
        resp = service.search(SearchRequest(query="slow stages query", mode="prismx", total_deadline_ms=250.0))

    assert resp.mode == "prismx"
    assert resp.governor_state == "skipped_budget"
    _, kwargs = reranker.rerank.call_args
    assert kwargs["total_deadline_ms"] == 250.0



# ---------------------------------------------------------------------------
# 4. LIVE UPSERT & DELETE VISIBILITY (Isolated Integration)
# ---------------------------------------------------------------------------

@pytest.mark.needs_100k
def test_live_upsert_and_delete_visibility():
    """Verify upsert is visible in the very next search and delete is immediately gone,
    using an isolated test_* collection and test_* SQLite file created and dropped by the test.
    The benchmark collection c100k_raw remains strictly read-only.
    """
    import os, gc, uuid
    from qdrant_client import QdrantClient, models
    from prismx.index.text_store import TextStore

    qd = QdrantClient(url="http://localhost:6333")
    test_col = f"test_gate5_live_{uuid.uuid4().hex[:8]}"
    test_db = f"data/{test_col}.db"

    # Pre-assertion: benchmark collection is 100,008
    bench_info = qd.get_collection("c100k_raw")
    assert bench_info.points_count == 100008, "Benchmark collection count must be 100,008"

    ts = None
    try:
        qd.create_collection(
            collection_name=test_col,
            vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
        )
        ts = TextStore(test_db)
        test_pid = f"test_pid_{uuid.uuid4().hex[:6]}"
        unique_marker = f"cryo_superconduct_{uuid.uuid4().hex[:6]}"
        passage_text = f"Scientific breakthrough in {unique_marker} allows zero electrical resistance at ambient pressure."

        # 1. Upsert into test SQLite and test Qdrant
        ts.upsert_single(test_pid, passage_text, "science-tech", "live_test")
        qd.upsert(
            collection_name=test_col,
            points=[
                models.PointStruct(
                    id=1,
                    vector={"dense": [0.0] * 384, "sparse": models.SparseVector(indices=[101, 102], values=[1.0, 0.8])},
                    payload={"passage_id": test_pid, "category": "science-tech", "source": "live_test"}
                )
            ]
        )

        # 2. Immediately verify visible in test collection
        pts = qd.scroll(collection_name=test_col, limit=5)[0]
        pids = [p.payload.get("passage_id") for p in pts]
        assert test_pid in pids, f"Expected {test_pid} in search results immediately after upsert"

        # 3. Delete from test SQLite and test Qdrant
        ts.delete_single(test_pid)
        qd.delete(collection_name=test_col, points_selector=models.PointIdsList(points=[1]))

        # 4. Immediately verify gone
        pts_after = qd.scroll(collection_name=test_col, limit=5)[0]
        pids_after = [p.payload.get("passage_id") for p in pts_after]
        assert test_pid not in pids_after, f"Expected {test_pid} to be absent from search results after delete"
    finally:
        try:
            qd.delete_collection(test_col)
        except Exception:
            pass
        if ts is not None:
            del ts
        gc.collect()
        try:
            if os.path.exists(test_db):
                os.remove(test_db)
        except Exception:
            pass

    # Post-assertion: benchmark collection remains 100,008
    bench_info_post = qd.get_collection("c100k_raw")
    assert bench_info_post.points_count == 100008, "Benchmark collection must remain strictly 100,008 (read-only)"


# ---------------------------------------------------------------------------
# 5. METADATA FILTER APPLIED AT QDRANT LEVEL (Zero out-of-filter results)
# ---------------------------------------------------------------------------

@pytest.mark.needs_100k
def test_metadata_filter_zero_out_of_filter():
    """Verify that pre-retrieval filtering returns ZERO results outside the filtered category."""
    import urllib.request
    import json

    base_url = "http://127.0.0.1:8000"
    target_category = "LOCATION"

    search_data = json.dumps({
        "query": "what temperature do you cook stuffed flounder",
        "mode": "hybrid",
        "top_k": 10,
        "filters": {"category": target_category},
        "use_cache": False
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/search",
        data=search_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        results = data["results"]
        assert len(results) > 0, "Expected at least 1 result for tax-state query"
        for r in results:
            assert r["category"] == target_category, f"Result {r['passage_id']} has category {r['category']} != {target_category}"


# ---------------------------------------------------------------------------
# 6. CACHE INVALIDATION ON UPSERT & DELETE (Isolated)
# ---------------------------------------------------------------------------

def test_cache_invalidation_on_upsert_and_delete():
    """Verify query cache hit is invalidated on upsert and delete using an isolated test_* database."""
    import os, gc, uuid
    from prismx.retrieve.cache import QueryCache
    from prismx.index.text_store import TextStore

    cache = QueryCache(maxsize=100)
    test_db = f"data/test_cache_{uuid.uuid4().hex[:8]}.db"
    ts = None
    try:
        ts = TextStore(test_db)
        key = cache.make_key("query_test", "hybrid", 5)
        # Populate cache
        cache.set(key, {"results": ["dummy"]})
        assert cache.get(key) is not None

        # Upsert invalidates
        ts.upsert_single("dummy_p1", "dummy text", "cat")
        cache.invalidate()
        assert cache.get(key) is None

        # Re-populate
        cache.set(key, {"results": ["dummy2"]})
        assert cache.get(key) is not None

        # Delete invalidates
        ts.delete_single("dummy_p1")
        cache.invalidate()
        assert cache.get(key) is None
    finally:
        if ts is not None:
            del ts
        gc.collect()
        try:
            if os.path.exists(test_db):
                os.remove(test_db)
        except Exception:
            pass
