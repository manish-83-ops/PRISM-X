"""One-command latency benchmark reproduction script for Gate 13 Addendum Item 4j.

Runs 100 consecutive queries against the FastAPI server (http://127.0.0.1:8000/search)
and outputs p50, p95, and p99 for:
Row 1: Cold start / no warm-up (first 100 consecutive queries without discarding any).
Row 2: 20 warm-up queries discarded (100 benchmark queries evaluated).

Usage:
    python scripts/reproduce_latency_100.py [--mode hybrid|dense|prismx]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_URL = "http://127.0.0.1:8000/search"
BENCH_PATH = REPO_ROOT / "data" / "c100k_raw" / "bench_raw_100.json"


def run_queries(queries: list[dict], mode: str, count: int, top_k: int = 5) -> list[float]:
    latencies_ms: list[float] = []
    for idx in range(count):
        q_item = queries[idx % len(queries)]
        q_text = q_item["query"]
        payload = {
            "query": q_text,
            "mode": mode,
            "top_k": top_k,
            "use_cache": False,
        }
        t0 = time.perf_counter()
        resp = requests.post(SEARCH_URL, json=payload, timeout=30)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            latencies_ms.append(wall_ms)
        else:
            print(f"Warning: query {idx+1} failed with status {resp.status_code}: {resp.text}")
    return latencies_ms


def compute_percentiles(latencies: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(latencies, dtype=float)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    return round(p50, 2), round(p95, 2), round(p99, 2)


def main():
    parser = argparse.ArgumentParser(description="PRISMX 100-Query Latency Reproduction")
    parser.add_argument("--mode", choices=["hybrid", "dense", "prismx"], default="hybrid", help="Retrieval mode to benchmark")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved candidates")
    args = parser.parse_args()

    if not BENCH_PATH.exists():
        print(f"Error: BENCH dataset not found at {BENCH_PATH}")
        sys.exit(1)

    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    print("=" * 80)
    print(f"PRISMX LATENCY BENCHMARK REPRODUCTION (100 CONSECUTIVE QUERIES, MODE: {args.mode.upper()})")
    print(f"Benchmark Dataset: {BENCH_PATH} ({len(bench_queries)} queries)")
    print(f"Server Endpoint:   {SEARCH_URL}")
    print("=" * 80)

    # Check server availability
    try:
        r = requests.get("http://127.0.0.1:8000/ready", timeout=5)
        if r.status_code != 200 or not r.json().get("ready", True):
            print(f"Warning: /ready returned {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Error: Cannot connect to server at {SEARCH_URL}: {e}")
        print("Please start the server first: python scripts/run_server.py")
        sys.exit(1)

    print("\n[Phase 1] Executing first 100 consecutive queries (no warmup discarded)...")
    lat_no_warmup = run_queries(bench_queries, args.mode, 100, args.top_k)
    p50_raw, p95_raw, p99_raw = compute_percentiles(lat_no_warmup)

    print("[Phase 2] Executing 20 warm-up queries (to be discarded)...")
    _ = run_queries(bench_queries, args.mode, 20, args.top_k)

    print("[Phase 3] Executing 100 consecutive queries (warmup discarded)...")
    lat_with_warmup = run_queries(bench_queries, args.mode, 100, args.top_k)
    p50_warm, p95_warm, p99_warm = compute_percentiles(lat_with_warmup)

    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK REPRODUCTION RESULTS")
    print("=" * 80)
    print(f"| Evaluation Protocol | N Queries | p50 (ms) | p95 (ms) | p99 (ms) | SLA Status (<300ms) |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(f"| First 100 Queries (Fresh / No Warmup Discarded) | 100 | {p50_raw} | {p95_raw} | {p99_raw} | {'PASS' if p95_raw < 300.0 else 'FAIL'} |")
    print(f"| 100 Queries (20 Warmups Discarded)              | 100 | {p50_warm} | {p95_warm} | {p99_warm} | {'PASS' if p95_warm < 300.0 else 'FAIL'} |")
    print("=" * 80)


if __name__ == "__main__":
    main()
