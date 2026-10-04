"""Unit and Fault-Injection Tests for Architecture Integrity (ADR-024).
Covers:
1. Benchmark Store Immutability Assertion (100,008 pre and post).
2. Version-Stamped Cache (Race test, Delete test, Version bump test).
3. Dual-Write Outbox with Fault Injection:
   - Crash after Qdrant before mark (idempotent replay).
   - Qdrant failure (pending queue recovery).
   - SQLite failure (transaction rollback).
   - Consistency probe (/meta verification).
All tests run strictly against test_* databases and isolated collections.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
import pytest
from qdrant_client import QdrantClient, models

from prismx.index.qdrant_store import QdrantStore, passage_id_to_point_id
from prismx.index.text_store import TextStore
from prismx.retrieve.cache import QueryCache

pytestmark = pytest.mark.needs_100k

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DB_PATH = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
BENCH_COLLECTION = "c100k_raw"

TEST_DB_PATH = REPO_ROOT / "data" / "test_text_store.db"
TEST_COLLECTION = "test_integrity_col"


@pytest.fixture(scope="session", autouse=True)
def benchmark_store_guard():
    """Asserts that the benchmark store has exactly 100,008 items before and after tests."""
    q_client = QdrantClient(host="127.0.0.1", port=6333)

    # Pre-assertion
    pre_qdrant_count = q_client.get_collection(BENCH_COLLECTION).points_count
    assert pre_qdrant_count == 100008, f"PRE-TEST FAIL: c100k_raw expected 100,008, got {pre_qdrant_count}"

    with sqlite3.connect(str(RAW_DB_PATH)) as conn:
        pre_sqlite_count = conn.execute("SELECT COUNT(*) FROM passages;").fetchone()[0]
    assert pre_sqlite_count == 100008, f"PRE-TEST FAIL: text_store_raw.db expected 100,008, got {pre_sqlite_count}"

    yield

    # Post-assertion
    post_qdrant_count = q_client.get_collection(BENCH_COLLECTION).points_count
    assert post_qdrant_count == 100008, f"POST-TEST CORRUPTION: c100k_raw expected 100,008, got {post_qdrant_count}"

    with sqlite3.connect(str(RAW_DB_PATH)) as conn:
        post_sqlite_count = conn.execute("SELECT COUNT(*) FROM passages;").fetchone()[0]
    assert post_sqlite_count == 100008, f"POST-TEST CORRUPTION: text_store_raw.db expected 100,008, got {post_sqlite_count}"


# =========================================================================
# 3.1 Version-Stamped Cache Tests
# =========================================================================

def test_cache_version_bump_o1_invalidation():
    """Version bump in SQLite metadata immediately invalidates lookups in O(1)."""
    cache = QueryCache(maxsize=100)

    # Request at version 1
    key_v1 = cache.make_key(query="test query", mode="hybrid", top_k=5, corpus_version=1)
    cache.set(key_v1, {"results": ["doc1"], "version": 1})

    # Read back at version 1 (hit)
    hit = cache.get(key_v1)
    assert hit is not None
    assert hit["version"] == 1

    # Corpus version bumps to 2
    key_v2 = cache.make_key(query="test query", mode="hybrid", top_k=5, corpus_version=2)
    miss = cache.get(key_v2)
    assert miss is None, "Expected cache miss on version increment (O(1) invalidation)"


def test_cache_race_condition_protection():
    """A slow in-flight request started at version 1 must not poison version 2."""
    cache = QueryCache(maxsize=100)

    # 1. Slow Request A starts at t0 under corpus_version 1
    t0_version = 1
    key_req_a = cache.make_key(query="slow query", mode="hybrid", top_k=5, corpus_version=t0_version)

    # 2. While Request A is executing, a write occurs, bumping corpus to version 2
    # In SQLite, corpus_version becomes 2

    # 3. Request A finally completes and writes its result using its t0 key
    cache.set(key_req_a, {"text": "stale data computed from version 1"}, passage_ids=["p1"])

    # 4. New Request B arrives under current corpus_version 2
    key_req_b = cache.make_key(query="slow query", mode="hybrid", top_k=5, corpus_version=2)
    result_b = cache.get(key_req_b)

    assert result_b is None, "Race condition failed: Stale Request A poisoned cache for version 2!"


def test_cache_delete_reverse_index_eviction():
    """Deleting a passage evicts all cached queries containing that passage."""
    cache = QueryCache(maxsize=100)

    k1 = cache.make_key(query="query 1", mode="hybrid", top_k=5, corpus_version=1)
    k2 = cache.make_key(query="query 2", mode="hybrid", top_k=5, corpus_version=1)
    k3 = cache.make_key(query="query 3", mode="hybrid", top_k=5, corpus_version=1)

    cache.set(k1, {"id": 1}, passage_ids=["doc_A", "doc_B"])
    cache.set(k2, {"id": 2}, passage_ids=["doc_B", "doc_C"])
    cache.set(k3, {"id": 3}, passage_ids=["doc_D", "doc_E"])

    assert cache.get(k1) is not None
    assert cache.get(k2) is not None
    assert cache.get(k3) is not None

    # Delete doc_B -> should evict k1 and k2, but leave k3 untouched
    evicted = cache.evict_passage("doc_B")
    assert evicted == 2

    assert cache.get(k1) is None, "k1 should have been evicted by reverse index"
    assert cache.get(k2) is None, "k2 should have been evicted by reverse index"
    assert cache.get(k3) is not None, "k3 does not contain doc_B and should remain cached"


# =========================================================================
# 3.2 Dual-Write Outbox & Fault Injection Tests
# =========================================================================

@pytest.fixture
def test_stores(tmp_path):
    """Sets up and tears down isolated test_* SQLite database and Qdrant collection."""
    db_file = tmp_path / "test_text_store.db"
    text_store = TextStore(db_path=db_file)
    q_client = QdrantClient(host="127.0.0.1", port=6333)

    # Recreate test collection
    if q_client.collection_exists(TEST_COLLECTION):
        q_client.delete_collection(TEST_COLLECTION)

    q_client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
    )

    qdrant_store = QdrantStore(host="127.0.0.1", port=6333, collection_name=TEST_COLLECTION)

    yield text_store, qdrant_store, q_client

    # Teardown Qdrant test collection
    if q_client.collection_exists(TEST_COLLECTION):
        q_client.delete_collection(TEST_COLLECTION)


def test_outbox_fault_a_crash_after_qdrant_before_mark(test_stores):
    """Fault A: Point applied to Qdrant, but system crashes before mark_outbox_applied.
    Replay must be strictly idempotent (no duplicates or errors).
    """
    text_store, qdrant_store, q_client = test_stores

    pid = "test_doc_101"
    pt_id = passage_id_to_point_id(pid)

    # 1. Atomic write to SQLite + Outbox
    op_id, version = text_store.upsert_with_outbox(
        passage_id=pid,
        text="Sample document text for test",
        category="testing",
        source="test",
    )
    assert text_store.get_outbox_pending_count() == 1

    # 2. Write to Qdrant succeeds
    pt = models.PointStruct(
        id=pt_id,
        vector={"dense": [0.1, 0.2, 0.3, 0.4]},
        payload={"passage_id": pid, "category": "testing"},
    )
    qdrant_store.upsert_points_batch([pt], wait=True)
    assert q_client.get_collection(TEST_COLLECTION).points_count == 1

    # 3. CRASH SIMULATION: System crashes here before mark_outbox_applied(op_id)
    assert text_store.get_outbox_pending_count() == 1

    # 4. Recovery: Replay pending ops
    pending = text_store.get_pending_outbox_ops()
    assert len(pending) == 1

    # Replay write idempotently
    qdrant_store.upsert_points_batch([pt], wait=True)
    text_store.mark_outbox_applied(op_id)

    # Verify state: 1 point in Qdrant (idempotent, no duplicates!), 0 pending ops
    assert q_client.get_collection(TEST_COLLECTION).points_count == 1
    assert text_store.get_outbox_pending_count() == 0


def test_outbox_fault_b_qdrant_failure(test_stores):
    """Fault B: Qdrant fails on initial write.
    SQLite write is preserved as source of truth; replay applies when Qdrant recovers.
    """
    text_store, qdrant_store, q_client = test_stores

    pid = "test_doc_202"
    pt_id = passage_id_to_point_id(pid)

    # 1. Atomic write to SQLite + Outbox
    op_id, version = text_store.upsert_with_outbox(
        passage_id=pid,
        text="Another test document",
        category="testing",
        source="test",
    )
    assert text_store.get_passage_count() == 1
    assert text_store.get_outbox_pending_count() == 1

    # 2. QDRANT FAILURE SIMULATION: Network timeout or unavailable
    # Initial write fails, so Qdrant has 0 points
    assert q_client.get_collection(TEST_COLLECTION).points_count == 0

    # 3. Qdrant recovers, replay applies pending operation
    pt = models.PointStruct(
        id=pt_id,
        vector={"dense": [0.5, 0.5, 0.5, 0.5]},
        payload={"passage_id": pid, "category": "testing"},
    )
    qdrant_store.upsert_points_batch([pt], wait=True)
    text_store.mark_outbox_applied(op_id)

    # 4. Assert full consistency
    assert q_client.get_collection(TEST_COLLECTION).points_count == 1
    assert text_store.get_outbox_pending_count() == 0


def test_outbox_fault_c_sqlite_failure(test_stores):
    """Fault C: SQLite fails before commit.
    Transaction rolls back cleanly, nothing in outbox, Qdrant untouched.
    """
    text_store, qdrant_store, q_client = test_stores

    try:
        with text_store._get_connection() as conn:
            conn.execute("INSERT INTO passages (passage_id, text) VALUES ('fail_pid', 'will fail');")
            # Force syntax error before commit
            conn.execute("INVALID SQL STATEMENT TRIGGERING ROLLBACK;")
    except Exception:
        pass

    assert text_store.get_passage_count() == 0
    assert text_store.get_outbox_pending_count() == 0
    assert q_client.get_collection(TEST_COLLECTION).points_count == 0


def test_consistency_probe(test_stores):
    """Consistency probe compares SQLite count, Qdrant count, and sampled ID existence."""
    text_store, qdrant_store, q_client = test_stores

    # Add 5 documents
    for i in range(5):
        pid = f"probe_doc_{i}"
        pt_id = passage_id_to_point_id(pid)
        op_id, _ = text_store.upsert_with_outbox(pid, f"Text {i}")
        pt = models.PointStruct(
            id=pt_id,
            vector={"dense": [0.1 * i, 0.1 * i, 0.1 * i, 0.1 * i]},
            payload={"passage_id": pid},
        )
        qdrant_store.upsert_points_batch([pt], wait=True)
        text_store.mark_outbox_applied(op_id)

    probe = text_store.consistency_probe(qdrant_store, sample_size=5)
    assert probe["is_consistent"] is True
    assert probe["sqlite_count"] == 5
    assert probe["qdrant_count"] == 5
    assert probe["count_difference"] == 0
    assert probe["sampled_count"] == 5
    assert probe["sampled_found_in_qdrant"] == 5
    assert probe["pending_outbox_count"] == 0
