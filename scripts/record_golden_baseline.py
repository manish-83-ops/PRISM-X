#!/usr/bin/env python3
"""
Record Golden Baseline for Gate 15 (A0)
Stores top-10 passage IDs and scores for 100 BENCH queries x dense / hybrid / prismx
into results/golden/golden_baseline_top10.json.
No timing claims.
"""

import json
import os
import sys
import time
import requests
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_PATH = REPO_ROOT / "data" / "c100k_raw" / "bench_raw_100.json"
OUT_DIR = REPO_ROOT / "results" / "golden"
OUT_FILE = OUT_DIR / "golden_baseline_top10.json"
ENDPOINT = "http://127.0.0.1:8000/search"


def main():
    if not BENCH_PATH.exists():
        print(f"[ERROR] Bench file not found at {BENCH_PATH}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    print(f"[*] Loaded {len(bench_queries)} BENCH queries from {BENCH_PATH}")
    print(f"[*] Target output: {OUT_FILE}")

    # Check server availability
    try:
        r = requests.get("http://127.0.0.1:8000/ready", timeout=5)
        if r.status_code != 200:
            print(f"[ERROR] /ready returned status {r.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Cannot connect to server at http://127.0.0.1:8000: {e}")
        sys.exit(1)

    modes = ["dense", "hybrid", "prismx"]
    results_by_query = []

    total_queries = len(bench_queries)
    for q_idx, q_item in enumerate(bench_queries, 1):
        q_id = q_item["query_id"]
        q_text = q_item["query"]

        q_record = {
            "query_id": q_id,
            "query": q_text,
            "modes": {},
        }

        for mode in modes:
            payload = {
                "query": q_text,
                "mode": mode,
                "top_k": 10,
                "use_cache": False,
            }
            resp = requests.post(ENDPOINT, json=payload, timeout=30)
            if resp.status_code != 200:
                print(f"[ERROR] Query {q_id} mode {mode} failed: {resp.status_code} {resp.text}")
                sys.exit(1)

            data = resp.json()
            items = []
            for r in data.get("results", []):
                items.append({
                    "rank": r.get("rank"),
                    "passage_id": r.get("passage_id"),
                    "score": round(float(r.get("score", 0.0)), 8),
                })

            mode_info = {
                "top10": items,
                "count": len(items),
            }
            if mode == "prismx":
                mode_info["candidates_scored"] = data.get("candidates_scored")
                mode_info["governor_state"] = data.get("governor_state")
                mode_info["stage_reached"] = data.get("stage_reached")
                mode_info["effective_mode"] = data.get("effective_mode")

            q_record["modes"][mode] = mode_info

        results_by_query.append(q_record)
        if q_idx % 20 == 0 or q_idx == total_queries:
            print(f"[*] Processed {q_idx}/{total_queries} queries...")

    output_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_dataset": "data/c100k_raw/bench_raw_100.json",
            "query_count": total_queries,
            "top_k": 10,
            "modes": modes,
            "purpose": "ADR-027 Golden Baseline before serving layer upgrades (A0)",
        },
        "queries": results_by_query,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[SUCCESS] Recorded golden baseline for {total_queries} queries into {OUT_FILE}")
    print(f"          File size: {os.path.getsize(OUT_FILE) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
