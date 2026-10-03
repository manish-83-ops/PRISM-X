"""PRISMX Search Service Unit Tests verifying PATCH-2 Decoupled Ordering and Inconsistency Tolerance."""

import pytest
from unittest.mock import MagicMock
from prismx.retrieve.service import SearchService
from prismx.schemas import SearchRequest, FilterParams

def test_decoupled_ordering_preservation():
    """Verify that SearchService preserves candidate rank order regardless of SQLite return order (ADR-007)."""
    dense_retriever = MagicMock()
    hybrid_retriever = MagicMock()
    text_store = MagicMock()
    qdrant_store = MagicMock()

    config = {
        "retrieval": {
            "candidate_depth": 20,
            "fusion": {"method": "weighted", "alpha": 0.7, "rrf_k": 60}
        }
    }

    service = SearchService(
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        text_store=text_store,
        qdrant_store=qdrant_store,
        config=config,
    )

    # Mock dense retriever candidate ordering: [pid_10, pid_20, pid_30]
    dense_retriever.retrieve.return_value = (
        [
            {"passage_id": "10", "dense_score": 0.95, "dense_rank": 1},
            {"passage_id": "20", "dense_score": 0.90, "dense_rank": 2},
            {"passage_id": "30", "dense_score": 0.85, "dense_rank": 3},
        ],
        5.0, # enc_ms
        10.0, # dense_ms
    )

    # SQLite returns passages in completely reversed or scrambled order
    text_store.get_passages_by_ids.return_value = {
        "30": {"passage_id": "30", "text": "Text for 30", "category": "cat3", "source": "src3"},
        "10": {"passage_id": "10", "text": "Text for 10", "category": "cat1", "source": "src1"},
        "20": {"passage_id": "20", "text": "Text for 20", "category": "cat2", "source": "src2"},
    }
    text_store.get_meta.return_value = 1

    req = SearchRequest(query="test query", mode="dense", top_k=3)
    resp = service.search(req)

    # Output ordering MUST strictly be 10, then 20, then 30
    assert len(resp.results) == 3
    assert resp.results[0].passage_id == "10"
    assert resp.results[0].rank == 1
    assert resp.results[0].text == "Text for 10"

    assert resp.results[1].passage_id == "20"
    assert resp.results[1].rank == 2
    assert resp.results[1].text == "Text for 20"

    assert resp.results[2].passage_id == "30"
    assert resp.results[2].rank == 3
    assert resp.results[2].text == "Text for 30"


def test_inconsistency_tolerance_and_backfilling():
    """Verify that if a passage exists in Qdrant but missing from SQLite, it skips and backfills (ADR-007)."""
    dense_retriever = MagicMock()
    hybrid_retriever = MagicMock()
    text_store = MagicMock()
    qdrant_store = MagicMock()

    config = {
        "retrieval": {
            "candidate_depth": 20,
            "fusion": {"method": "weighted", "alpha": 0.7, "rrf_k": 60}
        }
    }

    service = SearchService(
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        text_store=text_store,
        qdrant_store=qdrant_store,
        config=config,
    )

    # Candidates: pid_10, pid_20 (corrupted/missing in SQLite), pid_30
    dense_retriever.retrieve.return_value = (
        [
            {"passage_id": "10", "dense_score": 0.95, "dense_rank": 1},
            {"passage_id": "20", "dense_score": 0.90, "dense_rank": 2},
            {"passage_id": "30", "dense_score": 0.85, "dense_rank": 3},
        ],
        5.0,
        10.0,
    )

    # SQLite is missing passage "20"
    text_store.get_passages_by_ids.return_value = {
        "10": {"passage_id": "10", "text": "Text for 10", "category": "cat1", "source": "src1"},
        "30": {"passage_id": "30", "text": "Text for 30", "category": "cat3", "source": "src3"},
    }
    text_store.get_meta.return_value = 1

    req = SearchRequest(query="test query", mode="dense", top_k=2)
    resp = service.search(req)

    # Result should backfill "30" into rank 2, and track inconsistency
    assert len(resp.results) == 2
    assert resp.results[0].passage_id == "10"
    assert resp.results[1].passage_id == "30"
    assert resp.results[1].rank == 2
    assert service.inconsistency_count == 1
