"""Gate 2 Verification and Ingestion Integrity Tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import pytest
from qdrant_client import QdrantClient, models

from prismx.config import load_config
from prismx.index.qdrant_store import passage_id_to_point_id
from prismx.index.text_store import TextStore


def test_zero_label_leakage():
    """Verify strict split partitioning, label leakage absence, and near-duplicate audit."""
    tune_path = Path("data/manifests/split_tune.json")
    test_path = Path("data/manifests/split_test.json")
    bench_path = Path("data/manifests/split_bench.json")
    ragas_path = Path("data/manifests/split_ragas.json")

    assert tune_path.exists(), "split_tune.json must exist"
    assert test_path.exists(), "split_test.json must exist"
    assert bench_path.exists(), "split_bench.json must exist"
    assert ragas_path.exists(), "split_ragas.json must exist"

    with open(tune_path, "r", encoding="utf-8") as f:
        tune_q = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_q = json.load(f)
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_q = json.load(f)
    with open(ragas_path, "r", encoding="utf-8") as f:
        ragas_q = json.load(f)

    tune_ids = {str(q["query_id"]) for q in tune_q}
    test_ids = {str(q["query_id"]) for q in test_q}
    bench_ids = {str(q["query_id"]) for q in bench_q}
    ragas_ids = {str(q["query_id"]) for q in ragas_q}

    # Strict partition counts
    assert len(tune_ids) == 500, f"Expected 500 TUNE queries, got {len(tune_ids)}"
    assert len(test_ids) == 500, f"Expected 500 TEST queries, got {len(test_ids)}"
    assert len(bench_ids) == 100, f"Expected 100 BENCH queries, got {len(bench_ids)}"
    assert len(ragas_ids) == 100, f"Expected 100 RAGAS queries, got {len(ragas_ids)}"

    # TUNE and TEST must be strictly disjoint
    assert tune_ids.isdisjoint(test_ids), "TUNE and TEST query splits MUST be completely disjoint"

    # Verify corpus contains NO label annotations
    corpus_path = Path("data/corpus_100k.jsonl")
    assert corpus_path.exists()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            record = json.loads(line)
            assert "_is_gold" not in record
            assert "is_gold" not in record
            assert "qrels" not in record
            assert "query_id" not in record

    # Near-duplicates audit manifest exists (PATCH-4)
    near_dup_path = Path("data/manifests/near_duplicates_manifest.json")
    assert near_dup_path.exists(), "near_duplicates_manifest.json must exist"


@pytest.mark.integration
def test_text_store_schema_and_metadata():
    """Verify SQLite schema, running counts, and PATCH-1 drift tracking."""
    cfg = load_config()
    db_path = Path(cfg["sqlite"]["db_path"])
    assert db_path.exists(), f"SQLite DB does not exist at {db_path}"

    text_store = TextStore(str(db_path))
    stats = text_store.get_stats()
    text_store.close()

    assert stats["n_docs"] == 100000, f"Expected 100,000 documents in SQLite, got {stats['n_docs']}"
    assert stats["total_doc_len"] > 0, "total_doc_len must be positive"
    assert stats["avgdl_ref"] > 0.0, "avgdl_ref must be positive"
    assert stats["true_avgdl"] > 0.0, "true_avgdl must be positive"
    assert stats["drift"] < 0.10, f"Drift {stats['drift']} exceeds 10% threshold"
    assert not stats["drift_exceeds_threshold"], "drift_exceeds_threshold must be False at index creation"


@pytest.mark.integration
def test_qdrant_store_collection_integrity():
    """Verify Qdrant dual-vector configuration, IDF modifier, and point count."""
    cfg = load_config()
    client = QdrantClient(
        url=f"http://{cfg['qdrant']['host']}:{cfg['qdrant']['port']}",
        grpc_port=cfg["qdrant"]["grpc_port"],
        prefer_grpc=True,
    )
    col_name = cfg["qdrant"]["collection_name"]
    info = client.get_collection(col_name)

    assert info.points_count == 100000, f"Expected 100,000 points in Qdrant, got {info.points_count}"

    # Verify vector configs
    params = info.config.params
    assert "dense" in params.vectors, "Dense vector configuration missing"
    assert params.vectors["dense"].size == 384, "Dense dimension must be 384"
    assert params.vectors["dense"].distance == models.Distance.COSINE

    assert params.sparse_vectors is not None, "Sparse vector configuration missing"
    assert "bm25" in params.sparse_vectors, "Sparse BM25 configuration missing"
    assert params.sparse_vectors["bm25"].modifier == models.Modifier.IDF, "BM25 must use dynamic IDF modifier"

    # Verify payload schema
    payload_schema = info.payload_schema
    assert "category" in payload_schema, "Keyword index on 'category' missing"
    assert "source" in payload_schema, "Keyword index on 'source' missing"


@pytest.mark.integration
def test_round_trip_text_and_payload():
    """Sample 10 passages and verify round-trip fidelity across corpus, SQLite, and Qdrant."""
    cfg = load_config()
    db_path = Path(cfg["sqlite"]["db_path"])
    text_store = TextStore(str(db_path))
    client = QdrantClient(
        url=f"http://{cfg['qdrant']['host']}:{cfg['qdrant']['port']}",
        grpc_port=cfg["qdrant"]["grpc_port"],
        prefer_grpc=True,
    )
    col_name = cfg["qdrant"]["collection_name"]

    # Sample 10 lines from corpus
    corpus_path = Path("data/corpus_100k.jsonl")
    assert corpus_path.exists()

    samples = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 10000 == 500:
                samples.append(json.loads(line))
            if len(samples) == 10:
                break

    assert len(samples) == 10
    sample_ids = [s["passage_id"] for s in samples]
    hydrated = text_store.get_passages_by_ids(sample_ids)

    for item in samples:
        pid = item["passage_id"]
        # Verify SQLite text fidelity
        assert pid in hydrated, f"Passage {pid} not found in SQLite"
        assert hydrated[pid]["text"] == item["text"], f"Text mismatch for passage {pid}"

        # Verify Qdrant payload fidelity
        pt_id = passage_id_to_point_id(pid)
        pts = client.retrieve(collection_name=col_name, ids=[pt_id], with_payload=True)
        assert len(pts) == 1, f"Point {pt_id} not found in Qdrant"
        payload = pts[0].payload
        assert payload["passage_id"] == pid
        assert payload["category"] == hydrated[pid]["category"]
        assert payload["source"] == item["source"]

    text_store.close()
