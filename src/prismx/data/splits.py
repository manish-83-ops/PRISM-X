"""PRISMX Deterministic Query and Qrels Split Generator."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def load_dev_qrels(qrels_tsv_path: Path) -> dict[str, list[str]]:
    """Loads query-id to list of relevant passage IDs from dev.tsv."""
    qrels: dict[str, list[str]] = {}
    with open(qrels_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                qid, docid = str(row[0]).strip(), str(row[1]).strip()
                qrels.setdefault(qid, []).append(docid)
    return qrels

def load_dev_queries(queries_path: Path) -> dict[str, str]:
    """Loads query-id to query text from dev.jsonl.gz."""
    queries: dict[str, str] = {}
    with gzip.open(queries_path, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = str(item.get("query_id") or item.get("_id")).strip()
            text = (item.get("query") or item.get("text") or "").strip()
            queries[qid] = text
    return queries

def create_deterministic_splits(
    seed: int = 42,
    tune_count: int = 500,
    test_count: int = 500,
    bench_count: int = 100,
    raw_dir: Path | None = None,
    manifest_dir: Path | None = None,
) -> dict[str, Any]:
    """Generates strictly disjoint TUNE, TEST, BENCH, and RAGAS query splits."""
    raw_dir = raw_dir or (REPO_ROOT / "data" / "raw")
    manifest_dir = manifest_dir or (REPO_ROOT / "data" / "manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    qrels_path = raw_dir / "dev_qrels.tsv"
    queries_path = raw_dir / "dev_queries.jsonl.gz"

    if not qrels_path.is_file() or not queries_path.is_file():
        raise FileNotFoundError(f"Missing raw files: {qrels_path} or {queries_path}")

    qrels = load_dev_qrels(qrels_path)
    queries = load_dev_queries(queries_path)

    # Filter queries with >= 1 positive qrel
    valid_qids = [qid for qid in sorted(queries.keys()) if qid in qrels and len(qrels[qid]) > 0]
    total_valid = len(valid_qids)

    # Deterministic seeded shuffle
    rng = random.Random(seed)
    shuffled_qids = list(valid_qids)
    rng.shuffle(shuffled_qids)

    tune_qids = sorted(shuffled_qids[:tune_count])
    test_qids = sorted(shuffled_qids[tune_count : tune_count + test_count])

    # Disjointness check
    tune_set, test_set = set(tune_qids), set(test_qids)
    assert len(tune_set.intersection(test_set)) == 0, "TUNE and TEST splits MUST NOT overlap!"
    assert len(tune_qids) == tune_count, f"TUNE size must be {tune_count}"
    assert len(test_qids) == test_count, f"TEST size must be {test_count}"

    # BENCH = first bench_count queries of a fixed shuffled order of TEST
    bench_rng = random.Random(seed + 1)
    shuffled_test = list(test_qids)
    bench_rng.shuffle(shuffled_test)
    bench_qids = shuffled_test[:bench_count]
    ragas_qids = list(bench_qids)  # RAGAS = identical to BENCH

    # Build split records
    def build_records(qid_list: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "query_id": qid,
                "query": queries[qid],
                "gold_passage_ids": qrels[qid],
            }
            for qid in qid_list
        ]

    tune_records = build_records(tune_qids)
    test_records = build_records(test_qids)
    bench_records = build_records(bench_qids)
    ragas_records = build_records(ragas_qids)

    # Save individual split JSON files
    for name, records in [
        ("split_tune.json", tune_records),
        ("split_test.json", test_records),
        ("split_bench.json", bench_records),
        ("split_ragas.json", ragas_records),
    ]:
        out_path = manifest_dir / name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    # Master manifest
    manifest = {
        "dataset_name": "Tevatron/msmarco-passage + BeIR/msmarco-qrels",
        "total_dev_queries_with_qrels": total_valid,
        "seed": seed,
        "tune_count": len(tune_records),
        "test_count": len(test_records),
        "bench_count": len(bench_records),
        "ragas_count": len(ragas_records),
        "gold_passages_in_tune": len({d for r in tune_records for d in r["gold_passage_ids"]}),
        "gold_passages_in_test": len({d for r in test_records for d in r["gold_passage_ids"]}),
        "total_gold_passages_eval": len({d for r in tune_records + test_records for d in r["gold_passage_ids"]}),
        "hashes": {
            "split_tune_sha256": hashlib.sha256((manifest_dir / "split_tune.json").read_bytes()).hexdigest(),
            "split_test_sha256": hashlib.sha256((manifest_dir / "split_test.json").read_bytes()).hexdigest(),
            "split_bench_sha256": hashlib.sha256((manifest_dir / "split_bench.json").read_bytes()).hexdigest(),
            "split_ragas_sha256": hashlib.sha256((manifest_dir / "split_ragas.json").read_bytes()).hexdigest(),
        },
    }

    manifest_path = manifest_dir / "splits_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest

if __name__ == "__main__":
    m = create_deterministic_splits()
    print("Splits successfully generated:")
    print(f"  Total valid dev queries: {m['total_dev_queries_with_qrels']}")
    print(f"  TUNE queries: {m['tune_count']} (gold docs: {m['gold_passages_in_tune']})")
    print(f"  TEST queries: {m['test_count']} (gold docs: {m['gold_passages_in_test']})")
    print(f"  Total unique gold docs in TUNE+TEST: {m['total_gold_passages_eval']}")
    print(f"  BENCH / RAGAS queries: {m['bench_count']}")
