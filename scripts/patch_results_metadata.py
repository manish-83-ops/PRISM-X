"""Standardize metadata fields in results JSON files according to Gate 5.1 rules.
Rules: Each file must carry:
- commit
- date
- N
- seed
- config
- judge
- path_used
"""
import json
from pathlib import Path

COMMIT = "950bc5d6dbec421266e9cd78934dba2cac533224"
CONFIG_HASH = "64e95cabb1a1fd58e1ff021ff16043924b7eef86637d9e4defd1c0b5c7c4d2fd"

METADATA_MAP = {
    "results/ingest_stats.json": {
        "commit": COMMIT,
        "date": "2026-10-03 15:48:58",
        "N": 100000,
        "seed": 42,
        "config": {"config_hash": "3b06508a5c6dc296663e0547331da83b6b5996afa6a21d510fcbd84cb64cdd95"},
        "judge": "none",
        "path_used": "offline_ingest"
    },
    "results/phase1/metrics.json": {
        "commit": COMMIT,
        "date": "2026-10-03 18:20:00",
        "N": 100,
        "seed": 42,
        "config": {"mode": "dense", "top_k": 10},
        "judge": "deterministic_gold_labels",
        "path_used": "HTTP API (/search)"
    },
    "results/phase1/latency_benchmark.json": {
        "commit": COMMIT,
        "date": "2026-10-03 18:20:00",
        "N": 100,
        "seed": 42,
        "config": {"mode": "dense"},
        "judge": "none",
        "path_used": "HTTP API (/search)"
    },
    "results/phase2/metrics.json": {
        "commit": COMMIT,
        "date": "2026-10-03 18:20:00",
        "N": 100,
        "seed": 42,
        "config": {"mode": "hybrid", "alpha": 0.8, "method": "weighted"},
        "judge": "deterministic_gold_labels",
        "path_used": "HTTP API (/search)"
    },
    "results/phase2/benchmark_summary.json": {
        "commit": COMMIT,
        "date": "2026-10-03 18:20:00",
        "N": 100,
        "seed": 42,
        "config": {"mode": "hybrid", "alpha": 0.8},
        "judge": "none",
        "path_used": "HTTP API (/search)"
    },
    "results/phase3/metrics.json": {
        "commit": COMMIT,
        "date": "2026-10-03 19:14:43",
        "N": 100,
        "seed": 42,
        "config": {"config_hash": CONFIG_HASH, "rerank_depth": 10},
        "judge": "deterministic_gold_labels",
        "path_used": "HTTP API (/search)"
    },
    "results/phase3/latency_summary.json": {
        "commit": COMMIT,
        "date": "2026-10-03 19:18:27",
        "N": 100,
        "seed": 42,
        "config": {"rerank_k": 10, "deadline_ms": 200.0, "torch_threads": 8},
        "judge": "none",
        "path_used": "HTTP API (/search)"
    },
    "results/phase3/benchmark_summary.json": {
        "commit": COMMIT,
        "date": "2026-10-03 18:51:25",
        "N": 100,
        "seed": 42,
        "config": {"rerank_k": 10, "deadline_ms": 200.0},
        "judge": "none",
        "path_used": "HTTP API (/search)"
    },
    "results/ragas/frozen25_ragas_summary.json": {
        "commit": COMMIT,
        "date": "2026-10-03 19:30:00",
        "N": 25,
        "seed": 42,
        "config": {"rule": "ADR-014 MS MARCO reference answers"},
        "judge": "allam-2-7b",
        "path_used": "HTTP API (/search) + LLM Judge"
    },
    "results/ragas/frozen25_checkpoint.json": {
        "commit": COMMIT,
        "date": "2026-10-03 19:30:00",
        "N": 25,
        "seed": 42,
        "config": {"rule": "ADR-014 MS MARCO reference answers"},
        "judge": "allam-2-7b",
        "path_used": "HTTP API (/search) + LLM Judge"
    },
    "results/stress_test/stress_test_summary.json": {
        "commit": COMMIT,
        "date": "2026-10-03 20:21:00",
        "N": 100,
        "seed": 42,
        "config": {"collection": "c100k_hard", "distractors": 2887},
        "judge": "deterministic_gold_labels",
        "path_used": "HTTP API (/search)"
    },
    "results/stress_test/stress_test_ragas.json": {
        "commit": COMMIT,
        "date": "2026-10-03 20:21:00",
        "N": 25,
        "seed": 42,
        "config": {"collection": "c100k_hard"},
        "judge": "allam-2-7b",
        "path_used": "HTTP API (/search) + LLM Judge"
    },
    "results/candidate_movement_analysis.json": {
        "commit": COMMIT,
        "date": "2026-10-03 19:46:00",
        "N": 100,
        "seed": 42,
        "config": {"depth": 10, "alpha": 0.8},
        "judge": "none",
        "path_used": "HTTP API (/search)"
    }
}

for rel_path, meta in METADATA_MAP.items():
    p = Path(rel_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in meta.items():
            if k not in data:
                data[k] = v
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {rel_path} with metadata")
    else:
        print(f"Skipped {rel_path} (does not exist)")
