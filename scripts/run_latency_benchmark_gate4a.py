"""Gate 4A: Rigorous Latency Benchmark across Modes and Query Cache Workloads (NFR-3, C-05)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
import requests
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_URL = "http://127.0.0.1:8000/search"
OUT_DIR = REPO_ROOT / "results" / "phase3"


def run_benchmark_queries(
    query_list: list[dict],
    mode: str,
    top_k: int = 5,
    rerank_k: int = 30,
    use_cache: bool = False,
    scenario_label: str = "uncached",
) -> dict:
    query_records = []
    latencies = []

    for idx, item in enumerate(query_list, start=1):
        payload = {
            "query": item["query"],
            "mode": mode,
            "top_k": top_k,
            "rerank_k": rerank_k,
            "use_cache": use_cache,
        }
        t0 = time.perf_counter()
        resp = requests.post(SEARCH_URL, json=payload, timeout=30)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200, f"Query failed: {resp.text}"
        data = resp.json()
        s_lats = data.get("latency_ms", {})

        record = {
            "idx": idx,
            "query_id": item.get("query_id", str(idx)),
            "query": item["query"],
            "mode": mode,
            "scenario": scenario_label,
            "cache_hit": data.get("cache_hit", False),
            "client_wall_clock_ms": round(dt_ms, 2),
            "server_encode_ms": s_lats.get("encode", 0.0),
            "server_dense_ms": s_lats.get("dense", 0.0),
            "server_sparse_ms": s_lats.get("sparse", 0.0),
            "server_fusion_ms": s_lats.get("fusion", 0.0),
            "server_fetch_text_ms": s_lats.get("fetch_text", 0.0),
            "server_rerank_ms": s_lats.get("rerank", 0.0),
            "server_total_ms": s_lats.get("total", 0.0),
        }
        query_records.append(record)
        latencies.append(dt_ms)

    arr = np.array(latencies)
    return {
        "scenario": scenario_label,
        "mode": mode,
        "n_queries": len(latencies),
        "records": query_records,
        "p50_ms": round(float(np.percentile(arr, 50, method="linear")), 2),
        "p90_ms": round(float(np.percentile(arr, 90, method="linear")), 2),
        "p95_ms": round(float(np.percentile(arr, 95, method="linear")), 2),
        "p99_ms": round(float(np.percentile(arr, 99, method="linear")), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
    }


def save_csv(records: list[dict], filepath: Path) -> None:
    if not records:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    print("====================================================================")
    print("GATE 4A: RIGOROUS LATENCY BENCHMARK (NFR-3, C-05)")
    print("Per-Mode Uncached Runs + Dedicated Query Cache Workload Suite")
    print("====================================================================\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load BENCH split (100 queries)
    bench_path = REPO_ROOT / "data" / "manifests" / "split_bench.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    # 2. Invalidate cache before starting
    requests.post("http://127.0.0.1:8000/cache/invalidate")

    # 3. Discard 20 warm-up queries
    print("Step 0: Executing 20 warm-up queries (DISCARDED from percentiles)...")
    warmups = []
    for item in bench_queries[:20]:
        t0 = time.perf_counter()
        requests.post(SEARCH_URL, json={"query": item["query"], "mode": "hybrid", "top_k": 5, "use_cache": False})
        warmups.append((time.perf_counter() - t0) * 1000.0)
    print(f"Warm-ups complete: {len(warmups)} queries discarded. Mean warm-up latency: {np.mean(warmups):.2f} ms\n")

    benchmarks_summary = {}

    # 4. Mode 1: Dense Uncached (100 queries)
    print("Step 1: Benchmarking Mode: DENSE (Uncached, 100 queries)...")
    requests.post("http://127.0.0.1:8000/cache/invalidate")
    dense_res = run_benchmark_queries(bench_queries, mode="dense", top_k=5, use_cache=False, scenario_label="dense_uncached")
    save_csv(dense_res["records"], OUT_DIR / "latency_dense.csv")
    benchmarks_summary["dense"] = {k: v for k, v in dense_res.items() if k != "records"}
    print(f"  Dense: p50={dense_res['p50_ms']}ms | p90={dense_res['p90_ms']}ms | p95={dense_res['p95_ms']}ms | p99={dense_res['p99_ms']}ms | max={dense_res['max_ms']}ms")

    # 5. Mode 2: Hybrid Uncached (100 queries)
    print("\nStep 2: Benchmarking Mode: HYBRID (Uncached, 100 queries)...")
    requests.post("http://127.0.0.1:8000/cache/invalidate")
    hybrid_res = run_benchmark_queries(bench_queries, mode="hybrid", top_k=5, use_cache=False, scenario_label="hybrid_uncached")
    save_csv(hybrid_res["records"], OUT_DIR / "latency_hybrid_uncached.csv")
    benchmarks_summary["hybrid"] = {k: v for k, v in hybrid_res.items() if k != "records"}
    print(f"  Hybrid: p50={hybrid_res['p50_ms']}ms | p90={hybrid_res['p90_ms']}ms | p95={hybrid_res['p95_ms']}ms | p99={hybrid_res['p99_ms']}ms | max={hybrid_res['max_ms']}ms")

    # 6. Mode 3: Hybrid + Rerank (Uncached, 100 queries)
    print("\nStep 3: Benchmarking Mode: HYBRID+RERANK (Uncached, 100 queries)...")
    requests.post("http://127.0.0.1:8000/cache/invalidate")
    rerank_res = run_benchmark_queries(bench_queries, mode="hybrid_rerank", top_k=5, rerank_k=30, use_cache=False, scenario_label="hybrid_rerank_uncached")
    save_csv(rerank_res["records"], OUT_DIR / "latency_hybrid_rerank_uncached.csv")
    benchmarks_summary["hybrid_rerank"] = {k: v for k, v in rerank_res.items() if k != "records"}
    print(f"  Hybrid+Rerank: p50={rerank_res['p50_ms']}ms | p90={rerank_res['p90_ms']}ms | p95={rerank_res['p95_ms']}ms | p99={rerank_res['p99_ms']}ms | max={rerank_res['max_ms']}ms")

    # 7. Cache Workload Suite
    print("\n--------------------------------------------------------------------")
    print("STEP 4: QUERY CACHE WORKLOAD BENCHMARK SUITE")
    print("--------------------------------------------------------------------")

    # Workload A: All-Unique (0% repeated, cache enabled)
    requests.post("http://127.0.0.1:8000/cache/invalidate")
    print("Workload A: All-Unique Workload (100 distinct queries, 0% cache hits)...")
    cache_unique_res = run_benchmark_queries(bench_queries, mode="hybrid", top_k=5, use_cache=True, scenario_label="cache_all_unique")
    save_csv(cache_unique_res["records"], OUT_DIR / "latency_cache_workload_all_unique.csv")
    benchmarks_summary["cache_all_unique"] = {k: v for k, v in cache_unique_res.items() if k != "records"}
    print(f"  All-Unique: p50={cache_unique_res['p50_ms']}ms | p95={cache_unique_res['p95_ms']}ms")

    # Workload B: 30% Repeated Workload (70 unique, 30 repeats)
    # Warm first 30 queries into cache
    requests.post("http://127.0.0.1:8000/cache/invalidate")
    print("\nWorkload B: 30% Repeated Workload (70 unique queries + 30 repeated queries)...")
    workload_30_queries = []
    # Take first 30 queries, execute once to warm
    for item in bench_queries[:30]:
        requests.post(SEARCH_URL, json={"query": item["query"], "mode": "hybrid", "top_k": 5, "use_cache": True})
    # Now assemble 100 queries: 30 repeated + 70 remaining unique
    workload_30_queries.extend(bench_queries[:30])  # hits
    workload_30_queries.extend(bench_queries[30:100])  # misses
    cache_30_res = run_benchmark_queries(workload_30_queries, mode="hybrid", top_k=5, use_cache=True, scenario_label="cache_30_pct_repeated")
    save_csv(cache_30_res["records"], OUT_DIR / "latency_cache_workload_30pct_repeated.csv")
    benchmarks_summary["cache_30pct_repeated"] = {k: v for k, v in cache_30_res.items() if k != "records"}
    print(f"  30% Repeated: p50={cache_30_res['p50_ms']}ms | p95={cache_30_res['p95_ms']}ms")

    # Workload C: 100% Repeated Workload (100 identical queries replayed, 100% hits)
    print("\nWorkload C: 100% Repeated Workload (Best case, 100% cache hits)...")
    cache_100_res = run_benchmark_queries(bench_queries, mode="hybrid", top_k=5, use_cache=True, scenario_label="cache_100_pct_repeated")
    save_csv(cache_100_res["records"], OUT_DIR / "latency_cache_workload_100pct_repeated.csv")
    benchmarks_summary["cache_100pct_repeated"] = {k: v for k, v in cache_100_res.items() if k != "records"}
    print(f"  100% Repeated: p50={cache_100_res['p50_ms']}ms | p95={cache_100_res['p95_ms']}ms")

    # Save summary JSON
    summary_file = OUT_DIR / "benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": "Gate 4A Full Latency and Cache Suite",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "protocol": "D1 Client-side wall clock via HTTP API, D2 Linear interpolation percentiles",
                "warmup_queries_discarded": len(warmups),
                "warmup_mean_ms": round(float(np.mean(warmups)), 2),
                "sla_ceiling_ms": 300.0,
                "internal_target_ms": 250.0,
                "modes": benchmarks_summary,
            },
            f,
            indent=2,
        )

    print(f"\nSaved full latency benchmark summary to {summary_file}")


if __name__ == "__main__":
    main()
