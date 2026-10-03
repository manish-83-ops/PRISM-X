"""Gate 4B: Rigorous Latency Benchmark on Frozen Latency-Constrained Config (ADR-013).

Evaluates:
1. Mode Comparison on 100 BENCH queries (uncached):
   - Dense Baseline
   - Hybrid Optimized
   - Hybrid + MiniLM-L6 INT8 Rerank (K=10, 200ms deadline governor)
2. Query Cache Workload Comparison (on Hybrid+Rerank K=10):
   - All-unique queries (0% hit rate)
   - 30% repeated queries (realistic distribution)
   - 100% repeated queries (best case, 100% hit rate)
Outputs raw CSVs and latency_summary.json in results/phase3/.
"""

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
    rerank_k: int = 10,
    deadline_ms: float = 200.0,
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
            "deadline_ms": deadline_ms,
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
            "governor_state": data.get("governor_state", "normal"),
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
        "mean_ms": round(float(np.mean(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def write_csv(records: list[dict], filepath: Path):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    print("====================================================================")
    print("GATE 4B: OFFICIAL 100-QUERY LATENCY BENCHMARK (HTTP PATH, IDLE CPU)")
    print("Frozen config: MiniLM-L6 INT8, K=10, 200ms Deadline Governor")
    print("====================================================================\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bench_file = REPO_ROOT / "data" / "manifests" / "split_bench.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)
    assert len(bench_queries) == 100

    # 1. Warm-up
    print("Warming up server and cross-encoder on 5 queries...")
    for q in bench_queries[:5]:
        requests.post(SEARCH_URL, json={"query": q["query"], "mode": "hybrid_rerank", "rerank_k": 10, "use_cache": False})
    print("Warm-up complete.\n")

    # Invalidate cache before clean runs
    requests.post("http://127.0.0.1:8000/cache/invalidate")

    # 2. Benchmark Dense Baseline
    print("1. Benchmarking Dense Baseline (100 BENCH queries, uncached)...")
    dense_res = run_benchmark_queries(bench_queries, mode="dense", use_cache=False, scenario_label="dense_uncached")
    write_csv(dense_res["records"], OUT_DIR / "raw_latency_dense_bench100.csv")
    print(f"   p50={dense_res['p50_ms']}ms | p90={dense_res['p90_ms']}ms | p95={dense_res['p95_ms']}ms | p99={dense_res['p99_ms']}ms")

    # 3. Benchmark Hybrid
    print("\n2. Benchmarking Hybrid (Dense + BM25, 100 BENCH queries, uncached)...")
    hybrid_res = run_benchmark_queries(bench_queries, mode="hybrid", use_cache=False, scenario_label="hybrid_uncached")
    write_csv(hybrid_res["records"], OUT_DIR / "raw_latency_hybrid_bench100.csv")
    print(f"   p50={hybrid_res['p50_ms']}ms | p90={hybrid_res['p90_ms']}ms | p95={hybrid_res['p95_ms']}ms | p99={hybrid_res['p99_ms']}ms")

    # 4. Benchmark Hybrid + MiniLM-L6 INT8 Rerank (K=10, 200ms Governor)
    print("\n3. Benchmarking Hybrid + Rerank K=10 (100 BENCH queries, uncached)...")
    rerank_res = run_benchmark_queries(bench_queries, mode="hybrid_rerank", rerank_k=10, deadline_ms=200.0, use_cache=False, scenario_label="hybrid_rerank_uncached")
    write_csv(rerank_res["records"], OUT_DIR / "raw_latency_hybrid_rerank_bench100.csv")
    print(f"   p50={rerank_res['p50_ms']}ms | p90={rerank_res['p90_ms']}ms | p95={rerank_res['p95_ms']}ms | p99={rerank_res['p99_ms']}ms")

    # 5. Cache Workload 1: All-Unique Queries with Cache Enabled
    print("\n4. Benchmarking Query Cache: All-Unique Queries (0% hit rate, cache enabled)...")
    requests.post("http://127.0.0.1:8000/cache/invalidate")
    cache_unique_res = run_benchmark_queries(bench_queries, mode="hybrid_rerank", rerank_k=10, deadline_ms=200.0, use_cache=True, scenario_label="cache_unique_workload")
    write_csv(cache_unique_res["records"], OUT_DIR / "raw_latency_cache_all_unique.csv")
    print(f"   p50={cache_unique_res['p50_ms']}ms | p90={cache_unique_res['p90_ms']}ms | p95={cache_unique_res['p95_ms']}ms | p99={cache_unique_res['p99_ms']}ms")

    # 6. Cache Workload 2: 30% Repeated Queries Workload
    print("\n5. Benchmarking Query Cache: 30% Repeated Queries Workload...")
    import random
    rng = random.Random(42)
    workload_30 = []
    base_pool = bench_queries[:70]
    repeated_pool = bench_queries[:30]
    for _ in range(100):
        if rng.random() < 0.30:
            workload_30.append(rng.choice(repeated_pool))
        else:
            workload_30.append(rng.choice(base_pool))

    cache_30_res = run_benchmark_queries(workload_30, mode="hybrid_rerank", rerank_k=10, deadline_ms=200.0, use_cache=True, scenario_label="cache_30pct_workload")
    write_csv(cache_30_res["records"], OUT_DIR / "raw_latency_cache_30pct_repeated.csv")
    hits_30 = sum(1 for r in cache_30_res["records"] if r["cache_hit"])
    print(f"   p50={cache_30_res['p50_ms']}ms | p90={cache_30_res['p90_ms']}ms | p95={cache_30_res['p95_ms']}ms | Hits={hits_30}/100")

    # 7. Cache Workload 3: 100% Repeated Queries Workload
    print("\n6. Benchmarking Query Cache: 100% Repeated Queries Workload (Best Case)...")
    cache_100_res = run_benchmark_queries(bench_queries, mode="hybrid_rerank", rerank_k=10, deadline_ms=200.0, use_cache=True, scenario_label="cache_100pct_repeated_best_case")
    write_csv(cache_100_res["records"], OUT_DIR / "raw_latency_cache_100pct_repeated.csv")
    hits_100 = sum(1 for r in cache_100_res["records"] if r["cache_hit"])
    print(f"   p50={cache_100_res['p50_ms']}ms | p90={cache_100_res['p90_ms']}ms | p95={cache_100_res['p95_ms']}ms | Hits={hits_100}/100")

    # 8. Save overall summary
    summary_payload = {
        "benchmark": "Gate 4B Official HTTP Latency Benchmark (N=100 per scenario)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sla_target_p95_ms": 250.0,
        "sla_hard_limit_ms": 280.0,
        "rerank_config": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "quantization": "PyTorch dynamic INT8",
            "candidate_depth_k": 10,
            "max_length": 128,
            "torch_threads": 8,
            "deadline_ms": 200.0,
        },
        "mode_latency_uncached": {
            "dense": {k: v for k, v in dense_res.items() if k != "records"},
            "hybrid": {k: v for k, v in hybrid_res.items() if k != "records"},
            "hybrid_rerank_k10": {k: v for k, v in rerank_res.items() if k != "records"},
        },
        "cache_workloads": {
            "all_unique_0pct_hits": {k: v for k, v in cache_unique_res.items() if k != "records"},
            "repeated_30pct": {k: v for k, v in cache_30_res.items() if k != "records"},
            "repeated_100pct_best_case": {k: v for k, v in cache_100_res.items() if k != "records"},
        },
        "sla_status": {
            "hybrid_rerank_p95_ms": rerank_res["p95_ms"],
            "within_250ms_target": rerank_res["p95_ms"] <= 250.0,
            "within_280ms_hard_limit": rerank_res["p95_ms"] <= 280.0,
        },
    }

    with open(OUT_DIR / "latency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\nLatency summary and CSVs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
