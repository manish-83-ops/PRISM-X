"""Gate 1 Data, Splits, and Corpus Tests."""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_splits_manifest_validity():
    manifest_path = REPO_ROOT / "data" / "manifests" / "splits_manifest.json"
    assert manifest_path.is_file(), "Splits manifest must exist"
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    assert meta["tune_count"] == 500
    assert meta["test_count"] == 500
    assert meta["bench_count"] == 100
    assert meta["ragas_count"] == 100
    assert meta["total_gold_passages_eval"] == 1072

def test_tune_test_disjointness():
    tune_path = REPO_ROOT / "data" / "manifests" / "split_tune.json"
    test_path = REPO_ROOT / "data" / "manifests" / "split_test.json"
    with open(tune_path, "r", encoding="utf-8") as f:
        tune_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    tune_qids = {r["query_id"] for r in tune_data}
    test_qids = {r["query_id"] for r in test_data}
    assert len(tune_qids.intersection(test_qids)) == 0, "TUNE and TEST splits must be strictly disjoint!"

def test_bench_ragas_consistency():
    bench_path = REPO_ROOT / "data" / "manifests" / "split_bench.json"
    ragas_path = REPO_ROOT / "data" / "manifests" / "split_ragas.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    with open(ragas_path, "r", encoding="utf-8") as f:
        ragas_data = json.load(f)

    bench_qids = [r["query_id"] for r in bench_data]
    ragas_qids = [r["query_id"] for r in ragas_data]
    assert bench_qids == ragas_qids, "BENCH and RAGAS query sets must be identical"
    assert len(bench_qids) == 100

def test_corpus_manifest_and_leakage():
    corpus_manifest_path = REPO_ROOT / "data" / "manifests" / "corpus_manifest.json"
    assert corpus_manifest_path.is_file(), "Corpus manifest must exist"
    with open(corpus_manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_passages"] == 100000
    assert meta["gold_passages_count"] == 1072
    assert meta["gold_to_total_ratio"] == 0.01072

    # Check corpus file sample for absence of label leakage
    corpus_file = REPO_ROOT / "data" / "corpus_100k.jsonl"
    assert corpus_file.is_file(), "Corpus file must exist"
    with open(corpus_file, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert "passage_id" in first_line
        assert "text" in first_line
        assert "source" in first_line
        assert "_is_gold" not in first_line, "Gold label MUST NOT leak in payload/fields"
        assert "_hash" not in first_line
        assert "is_gold" not in first_line
