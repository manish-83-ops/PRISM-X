"""Gate 3 Sub-step 3c: Latency Benchmark conforming strictly to NFR-3 and C-05."""

import csv
import json
from pathlib import Path
import time
import requests
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

def run_100_queries(bench_queries, mode="hybrid", is_cached_run=False):
    search_url = "http://127.0.0.1:8000/search"
    query_records = []
    latencies = []

    for idx, item in enumerate(bench_queries, start=1):
        payload = {
            "query": item["query"],
            "mode": mode,
            "top_k": 5,
        }
        t0 = time.perf_counter()
        resp = requests.post(search_url, json=payload, timeout=20)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, f"Query failed: {resp.text}"

        data = resp.json()
        s_lats = data.get("latency_ms", {})
        query_records.append({
            "idx": idx,
            "query_id": item["query_id"],
            "query": item["query"],
            "client_wall_clock_ms": round(dt_ms, 2),
            "server_encode_ms": s_lats.get("encode", 0.0),
            "server_dense_ms": s_lats.get("dense", 0.0),
            "server_sparse_ms": s_lats.get("sparse", 0.0),
            "server_fusion_ms": s_lats.get("fusion", 0.0),
            "server_fetch_text_ms": s_lats.get("fetch_text", 0.0),
            "server_total_ms": s_lats.get("total", 0.0),
            "cached": is_cached_run,
        })
        latencies.append(dt_ms)

    arr = np.array(latencies)
    return {
        "records": query_records,
        "latencies": latencies,
        "p50_ms": round(float(np.percentile(arr, 50, method="linear")), 2),
        "p90_ms": round(float(np.percentile(arr, 90, method="linear")), 2),
        "p95_ms": round(float(np.percentile(arr, 95, method="linear")), 2),
        "p99_ms": round(float(np.percentile(arr, 99, method="linear")), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
    }

def main():
    bench_file = REPO_ROOT / "data" / "manifests" / "split_bench.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    assert len(bench_queries) == 100

    out_dir = REPO_ROOT / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    search_url = "http://127.0.0.1:8000/search"

    print("====================================================================")
    print("GATE 3c: STRICT QUERY LATENCY BENCHMARK (NFR-3, C-05)")
    print("====================================================================")

    # 1. Warm-up: 20 queries strictly discarded from percentiles
    print("Step 1: Running 20 sequential warm-up queries (DISCARDED from percentiles)...")
    warmup_lats = []
    for item in bench_queries[:20]:
        t0 = time.perf_counter()
        resp = requests.post(search_url, json={"query": item["query"], "mode": "hybrid", "top_k": 5}, timeout=15)
        warmup_lats.append((time.perf_counter() - t0) * 1000.0)
    warmup_mean = float(np.mean(warmup_lats))
    print(f"Warm-up complete: {len(warmup_lats)} queries discarded. Mean warm-up latency: {warmup_mean:.2f} ms\n")

    # 2. Uncached Benchmark (100 distinct queries)
    print("Step 2: Executing 100 distinct sequential BENCH queries (UNCACHED)...")
    uncached_res = run_100_queries(bench_queries, mode="hybrid", is_cached_run=False)

    # Save Uncached CSV
    csv_uncached = out_dir / "latency_hybrid_uncached.csv"
    with open(csv_uncached, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(uncached_res["records"][0].keys()))
        writer.writeheader()
        writer.writerows(uncached_res["records"])

    # Official latency_hybrid.csv (copy)
    with open(out_dir / "latency_hybrid.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(uncached_res["records"][0].keys()))
        writer.writeheader()
        writer.writerows(uncached_res["records"])

    print(f"Uncached Results: p50={uncached_res['p50_ms']}ms | p90={uncached_res['p90_ms']}ms | p95={uncached_res['p95_ms']}ms | p99={uncached_res['p99_ms']}ms | Max={uncached_res['max_ms']}ms")
    print(f"NFR-3 Compliance (<300ms): {'PASS' if uncached_res['p95_ms'] < 300.0 else 'FAIL'} (Internal Target <=250ms: {'PASS' if uncached_res['p95_ms'] <= 250.0 else 'FAIL'})")

    # 3. Cached Benchmark (same 100 queries repeated)
    print("\nStep 3: Executing 100 repeat sequential queries (CACHED / WARM OS PAGE CACHE)...")
    cached_res = run_100_queries(bench_queries, mode="hybrid", is_cached_run=True)

    csv_cached = out_dir / "latency_hybrid_cached.csv"
    with open(csv_cached, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cached_res["records"][0].keys()))
        writer.writeheader()
        writer.writerows(cached_res["records"])

    print(f"Cached Results:   p50={cached_res['p50_ms']}ms | p90={cached_res['p90_ms']}ms | p95={cached_res['p95_ms']}ms | p99={cached_res['p99_ms']}ms | Max={cached_res['max_ms']}ms")

    # 4. Summary JSON
    summary = {
        "benchmark": "NFR-3 / C-05 Hybrid Latency Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "D1 Client-side wall clock via HTTP API, D2 Linear interpolation percentiles",
        "warmup_queries_discarded": len(warmup_lats),
        "warmup_mean_ms": round(warmup_mean, 2),
        "uncached": {
            "n_queries": 100,
            "p50_ms": uncached_res["p50_ms"],
            "p90_ms": uncached_res["p90_ms"],
            "p95_ms": uncached_res["p95_ms"],
            "p99_ms": uncached_res["p99_ms"],
            "max_ms": uncached_res["max_ms"],
            "mean_ms": uncached_res["mean_ms"],
            "nfr3_pass_under_300ms": uncached_res["p95_ms"] < 300.0,
            "internal_target_le_250ms": uncached_res["p95_ms"] <= 250.0,
            "raw_csv": str(csv_uncached),
        },
        "cached": {
            "n_queries": 100,
            "p50_ms": cached_res["p50_ms"],
            "p90_ms": cached_res["p90_ms"],
            "p95_ms": cached_res["p95_ms"],
            "p99_ms": cached_res["p99_ms"],
            "max_ms": cached_res["max_ms"],
            "mean_ms": cached_res["mean_ms"],
            "raw_csv": str(csv_cached),
        },
    }

    summary_file = out_dir / "benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved full benchmark summary to {summary_file}")

if __name__ == "__main__":
    main()
