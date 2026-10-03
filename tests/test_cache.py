"""Unit tests for PRISMX Query Result Cache and Automatic Invalidation (ADR-012)."""

from unittest.mock import MagicMock
import pytest

from prismx.retrieve.cache import QueryCache
from prismx.retrieve.service import SearchService
from prismx.schemas import SearchRequest, UpsertRequest


@pytest.fixture
def mock_service():
    dense_retriever = MagicMock()
    hybrid_retriever = MagicMock()
    text_store = MagicMock()
    qdrant_store = MagicMock()

    config = {
        "retrieval": {
            "candidate_depth": 20,
            "fusion": {"method": "weighted", "alpha": 0.8, "rrf_k": 60},
        }
    }

    cache = QueryCache(maxsize=100)

    # Mock candidates
    dense_retriever.retrieve.return_value = (
        [{"passage_id": "p1", "dense_score": 0.9, "dense_rank": 1}],
        2.0,
        5.0,
    )
    hybrid_retriever.retrieve.return_value = (
        [{"passage_id": "p1", "score": 0.9, "dense_rank": 1, "bm25_rank": 1}],
        {"encode": 2.0, "dense": 5.0, "sparse": 3.0, "fusion": 1.0},
    )
    import numpy as np
    dense_retriever.encoder.encode_queries.return_value = np.zeros((1, 384))
    hybrid_retriever.tokenizer.compute_doc_sparse_vector.return_value = ([1], [1.0])
    hybrid_retriever.tokenizer.tokenize.return_value = ["new", "text"]

    text_store.get_passages_by_ids.return_value = {
        "p1": {"passage_id": "p1", "text": "Passage text 1", "category": "tech", "source": "msmarco"}
    }
    text_store.get_meta.return_value = 1
    text_store.upsert_single.return_value = 2
    text_store.delete_single.return_value = (True, 3)

    service = SearchService(
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        text_store=text_store,
        qdrant_store=qdrant_store,
        config=config,
        reranker=None,
        cache=cache,
    )
    return service, cache


def test_cache_hit_and_miss(mock_service):
    """Verify first query is a cache miss and second identical query is a cache hit."""
    service, cache = mock_service

    req = SearchRequest(query="machine learning", mode="hybrid", top_k=1, use_cache=True)

    # 1. First execution -> Cache miss
    resp1 = service.search(req)
    assert resp1.cache_hit is False
    assert len(resp1.results) == 1
    assert resp1.results[0].passage_id == "p1"
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 1
    assert cache.stats()["size"] == 1

    # 2. Second execution -> Cache hit
    resp2 = service.search(req)
    assert resp2.cache_hit is True
    assert len(resp2.results) == 1
    assert resp2.results[0].passage_id == "p1"
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_cache_invalidation_on_upsert(mock_service):
    """Verify upserting a passage clears the query cache to prevent stale reads."""
    service, cache = mock_service

    req = SearchRequest(query="neural networks", mode="hybrid", top_k=1)
    service.search(req)
    assert cache.stats()["size"] == 1

    # Upsert passage
    service.upsert_passage(UpsertRequest(passage_id="p_new", text="New text"))
    assert cache.stats()["size"] == 0

    # Next search is a fresh retrieval (miss)
    resp = service.search(req)
    assert resp.cache_hit is False
    assert cache.stats()["size"] == 1


def test_cache_invalidation_on_delete(mock_service):
    """Verify deleting a passage clears the query cache to prevent ghost results."""
    service, cache = mock_service

    req = SearchRequest(query="deep learning", mode="hybrid", top_k=1)
    service.search(req)
    assert cache.stats()["size"] == 1

    # Delete passage
    service.delete_passage("p1")
    assert cache.stats()["size"] == 0

    # Next search is a fresh retrieval (miss)
    resp = service.search(req)
    assert resp.cache_hit is False
    assert cache.stats()["size"] == 1


def test_cache_bypass_flag(mock_service):
    """Verify use_cache=False bypasses the cache completely."""
    service, cache = mock_service

    req1 = SearchRequest(query="gradient descent", mode="hybrid", top_k=1, use_cache=False)
    resp1 = service.search(req1)
    assert resp1.cache_hit is False
    assert cache.stats()["size"] == 0

    resp2 = service.search(req1)
    assert resp2.cache_hit is False
    assert cache.stats()["size"] == 0


def test_cache_key_differentiation(mock_service):
    """Verify different modes or filters produce distinct cache keys."""
    service, cache = mock_service

    req_hybrid = SearchRequest(query="transformers", mode="hybrid", top_k=1)
    req_dense = SearchRequest(query="transformers", mode="dense", top_k=1)

    service.search(req_hybrid)
    assert cache.stats()["size"] == 1

    service.search(req_dense)
    assert cache.stats()["size"] == 2
