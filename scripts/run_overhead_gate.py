#!/usr/bin/env python3
"""
PRISMX Gate 15 A6 Overhead Gate Benchmark.
Measures the serving layer overhead across 200 sequential default-mode (hybrid) requests:
- Baseline (features OFF: bypassed metrics, rate limiter, server-timing, structured queue logging)
- Upgraded (features ON: full Prometheus metrics, rate limiting, request-id, server-timing, async queue logging)

Hypothesis (ADR-027):
Delta p50 <= 1.0 ms
Delta p95 <= 2.0 ms
Mechanical Rule: If overhead gate fails, offending component is disabled by config default and documented.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
BENCH_PATH = Path("data/c100k_raw/bench_raw_100.json")
OUTPUT_PATH = Path("results/overhead_gate_results.json")
N_REQUESTS = 200


def measure_batch(queries: list[str], bypass: bool) -> list[float]:
    latencies = []
    headers = {"Content-Type": "application/json"}
    if bypass:
        headers["X-Bypass-Serving-Upgrades"] = "1"

    for idx, q in enumerate(queries, start=1):
        payload = {
            "query": q,
            "mode": "hybrid",
            "top_k": 10,
            "use_cache": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/search",
            data=data,
            headers=headers,
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            _ = resp.read()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        if idx % 50 == 0:
            print(f"  [{idx}/{len(queries)}] finished (recent: {dt_ms:.2f} ms)")
    return latencies


def run_benchmark():
    print("=" * 80)
    print("PRISMX GATE 15 A6: SERVING LAYER OVERHEAD BENCHMARK")
    print(f"Target: 200 sequential default-mode (hybrid) requests OFF vs ON")
    print("=" * 80)

    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    # Replicate 100 queries twice to form 200 sequential queries
    query_texts = [item["query"] for item in bench_queries]
    test_queries = (query_texts + query_texts)[:N_REQUESTS]

    # Warmup 5 requests
    print("[*] Running 5 warmup requests...")
    _ = measure_batch(test_queries[:5], bypass=False)

    print(f"\n[*] 1. Running {N_REQUESTS} requests with Serving Features OFF (Baseline)...")
    lat_off = measure_batch(test_queries, bypass=True)

    print(f"\n[*] 2. Running {N_REQUESTS} requests with Serving Features ON (Upgraded)...")
    lat_on = measure_batch(test_queries, bypass=False)

    def stats(vals: list[float]) -> dict[str, float]:
        s = sorted(vals)
        p50 = statistics.median(s)
        p95 = s[int(len(s) * 0.95)]
        mean = statistics.mean(s)
        return {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "mean": round(mean, 3),
            "min": round(min(s), 3),
            "max": round(max(s), 3),
        }

    off_stats = stats(lat_off)
    on_stats = stats(lat_on)

    delta_p50 = round(on_stats["p50"] - off_stats["p50"], 3)
    delta_p95 = round(on_stats["p95"] - off_stats["p95"], 3)

    p50_pass = delta_p50 <= 1.0
    p95_pass = delta_p95 <= 2.0
    overall_pass = p50_pass and p95_pass

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requests_evaluated": N_REQUESTS,
        "mode": "hybrid (default)",
        "features_off": off_stats,
        "features_on": on_stats,
        "deltas": {
            "delta_p50_ms": delta_p50,
            "delta_p95_ms": delta_p95,
        },
        "adr027_thresholds": {
            "max_delta_p50_ms": 1.0,
            "max_delta_p95_ms": 2.0,
        },
        "verdict": {
            "p50_gate": "PASS" if p50_pass else "FAIL",
            "p95_gate": "PASS" if p95_pass else "FAIL",
            "overall_status": "PASS" if overall_pass else "FAIL",
        }
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("OVERHEAD GATE RESULTS (ADR-027)")
    print("=" * 80)
    print(f"Features OFF (Baseline): p50 = {off_stats['p50']} ms | p95 = {off_stats['p95']} ms")
    print(f"Features ON  (Upgraded): p50 = {on_stats['p50']} ms | p95 = {on_stats['p95']} ms")
    print(f"Delta p50: {delta_p50:+.3f} ms (Target: <= 1.0 ms) -> {'PASS' if p50_pass else 'FAIL'}")
    print(f"Delta p95: {delta_p95:+.3f} ms (Target: <= 2.0 ms) -> {'PASS' if p95_pass else 'FAIL'}")
    print(f"Overall Status: {results['verdict']['overall_status']}")
    print(f"Saved results to: {OUTPUT_PATH}")
    print("=" * 80)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_benchmark())
