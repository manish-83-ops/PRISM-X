"""Gate 5.5: Rigorous Idle-Machine HTTP-Path Latency Benchmark for c100k_raw.

Evaluates:
1. 20 Warm-up queries (discarded from main stats and reported separately).
2. Mode Comparison on 100 BENCH queries (strictly sequential, uncached):
   - Dense Baseline
   - Hybrid (Dense + BM25, alpha=0.80)
   - PRISMX (Hybrid + INT8 Cross-Encoder Rerank K=10, 200ms budget, 250ms deadline)
3. Query Cache Workload Comparison (on PRISMX):
   - All-unique queries (0% hit rate)
   - 30% repeated queries (realistic distribution)

Telemetry logged:
- Machine power state (plugged in, active power scheme)
- Thread settings & library versions
- Git commit & canonical config hash
- 5x p95 latency outlier flagging

Outputs:
- results/c100k_raw/latency_benchmark.json
- Raw CSVs for all scenarios in results/c100k_raw/
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any
import numpy as np
import psutil
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_URL = "http://127.0.0.1:8000/search"
CACHE_INVALIDATE_URL = "http://127.0.0.1:8000/cache/invalidate"
META_URL = "http://127.0.0.1:8000/meta"
BENCH_PATH = REPO_ROOT / "data" / "c100k_raw" / "bench_raw_100.json"
OUT_DIR = REPO_ROOT / "results" / "c100k_raw"


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=str(REPO_ROOT))
        return res.stdout.strip()
    except Exception as e:
        return f"unknown ({e})"


def get_machine_power_state() -> dict[str, Any]:
    state = {
        "power_plugged": None,
        "battery_percent": None,
        "power_plan": "unknown"
    }
    try:
        b = psutil.sensors_battery()
        if b is not None:
            state["power_plugged"] = bool(b.power_plugged)
            state["battery_percent"] = float(b.percent)
    except Exception as e:
        state["battery_error"] = str(e)

    try:
        p = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        if p.returncode == 0:
            state["power_plan"] = p.stdout.strip()
    except Exception as e:
        state["power_plan_error"] = str(e)

    return state


def get_environment_info() -> dict[str, Any]:
    import importlib.metadata
    import torch
    import transformers
    import sentence_transformers
    import fastapi
    import uvicorn

    def pkg_ver(pkg_name: str, fallback: str = "unknown") -> str:
        try:
            return importlib.metadata.version(pkg_name)
        except Exception:
            return fallback

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "qdrant_client": pkg_ver("qdrant-client"),
        "fastapi": fastapi.__version__,
        "uvicorn": uvicorn.__version__,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "os": sys.platform,
    }


def run_sequential_queries(
    query_list: list[dict],
    mode: str,
    top_k: int = 5,
    rerank_k: int = 10,
    total_deadline_ms: float = 250.0,
    rerank_budget_ms: float = 200.0,
    use_cache: bool = False,
    scenario_label: str = "uncached",
) -> dict[str, Any]:
    records = []
    latencies = []

    for idx, item in enumerate(query_list, start=1):
        payload = {
            "query": item["query"],
            "mode": mode,
            "top_k": top_k,
            "rerank_k": rerank_k,
            "total_deadline_ms": total_deadline_ms,
            "rerank_budget_ms": rerank_budget_ms,
            "use_cache": use_cache,
        }
        t0 = time.perf_counter()
        resp = requests.post(SEARCH_URL, json=payload, timeout=60)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP Error {resp.status_code} on query {idx}: {resp.text}")

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
        records.append(record)
        latencies.append(dt_ms)

    arr = np.array(latencies, dtype=float)
    p50 = round(float(np.percentile(arr, 50, method="linear")), 2)
    p90 = round(float(np.percentile(arr, 90, method="linear")), 2)
    p95 = round(float(np.percentile(arr, 95, method="linear")), 2)
    p99 = round(float(np.percentile(arr, 99, method="linear")), 2)
    max_ms = round(float(np.max(arr)), 2)
    min_ms = round(float(np.min(arr)), 2)
    mean_ms = round(float(np.mean(arr)), 2)

    # Flag queries > 5x p95
    outlier_threshold = 5.0 * p95
    outliers = [
        {"idx": r["idx"], "query_id": r["query_id"], "query": r["query"], "client_wall_clock_ms": r["client_wall_clock_ms"]}
        for r in records if r["client_wall_clock_ms"] > outlier_threshold
    ]

    return {
        "scenario": scenario_label,
        "mode": mode,
        "n_queries": len(latencies),
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max_ms,
        "min_ms": min_ms,
        "mean_ms": mean_ms,
        "outlier_5x_p95_threshold_ms": round(outlier_threshold, 2),
        "outliers_count": len(outliers),
        "outliers": outliers,
        "records": records,
    }


def write_csv(records: list[dict], filepath: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    print("====================================================================")
    print("GATE 5.5: OFFICIAL IDLE-MACHINE HTTP LATENCY BENCHMARK (c100k_raw)")
    print("Sequential execution on 100 BENCH queries, 20 warm-ups discarded")
    print("====================================================================\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify server health and configuration
    print("Connecting to PRISMX API at http://127.0.0.1:8000...")
    meta_resp = requests.get(META_URL, timeout=10)
    assert meta_resp.status_code == 200, f"Server unreachable: {meta_resp.text}"
    meta = meta_resp.json()
    print(f"Server ready: {meta['point_count']:,} points, config hash: {meta['config_hash']}")
    print(f"Collection: {meta.get('point_count')} points, SQLite count: {meta.get('sqlite_count')}")

    # 2. Gather environment metadata
    git_commit = get_git_commit()
    power_state = get_machine_power_state()
    env_info = get_environment_info()
    print(f"Git commit: {git_commit}")
    print(f"Power: plugged={power_state['power_plugged']}, plan='{power_state['power_plan']}'")
    print(f"Torch threads: {env_info['torch_num_threads']} | Logical CPUs: {env_info['cpu_count_logical']}")

    # 3. Load 100 BENCH queries
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)
    assert len(bench_queries) == 100, f"Expected 100 queries, got {len(bench_queries)}"
    print(f"Loaded {len(bench_queries)} BENCH queries from {BENCH_PATH.name}\n")

    # 4. Phase 0: 20 Warm-up Queries (Discarded from main stats, reported separately)
    print(">>> Executing 20 Warm-up queries (mode=prismx, K=10, uncached)...")
    requests.post(CACHE_INVALIDATE_URL, timeout=10)
    warmup_queries = bench_queries[:20]
    warmup_res = run_sequential_queries(
        warmup_queries,
        mode="prismx",
        top_k=5,
        rerank_k=10,
        total_deadline_ms=250.0,
        rerank_budget_ms=200.0,
        use_cache=False,
        scenario_label="warmup_20_discarded",
    )
    write_csv(warmup_res["records"], OUT_DIR / "raw_latency_warmup_20.csv")
    print(f"   [Warm-up Discarded (N=20)] p50={warmup_res['p50_ms']}ms | p90={warmup_res['p90_ms']}ms | p95={warmup_res['p95_ms']}ms | max={warmup_res['max_ms']}ms | mean={warmup_res['mean_ms']}ms\n")

    # Invalidate cache before clean measurement
    requests.post(CACHE_INVALIDATE_URL, timeout=10)

    # 5. Benchmark 1: Dense Baseline (100 BENCH queries, uncached)
    print(">>> 1/5 Benchmarking Dense Baseline (100 queries, uncached)...")
    dense_res = run_sequential_queries(
        bench_queries,
        mode="dense",
        top_k=5,
        use_cache=False,
        scenario_label="dense_uncached",
    )
    write_csv(dense_res["records"], OUT_DIR / "raw_latency_dense_bench100.csv")
    print(f"   Dense:  p50={dense_res['p50_ms']}ms | p90={dense_res['p90_ms']}ms | p95={dense_res['p95_ms']}ms | p99={dense_res['p99_ms']}ms | max={dense_res['max_ms']}ms (Outliers > 5x p95: {dense_res['outliers_count']})")

    # Invalidate cache
    requests.post(CACHE_INVALIDATE_URL, timeout=10)

    # 6. Benchmark 2: Hybrid Baseline (100 BENCH queries, uncached)
    print("\n>>> 2/5 Benchmarking Hybrid (Dense + BM25, alpha=0.80, 100 queries, uncached)...")
    hybrid_res = run_sequential_queries(
        bench_queries,
        mode="hybrid",
        top_k=5,
        use_cache=False,
        scenario_label="hybrid_uncached",
    )
    write_csv(hybrid_res["records"], OUT_DIR / "raw_latency_hybrid_bench100.csv")
    print(f"   Hybrid: p50={hybrid_res['p50_ms']}ms | p90={hybrid_res['p90_ms']}ms | p95={hybrid_res['p95_ms']}ms | p99={hybrid_res['p99_ms']}ms | max={hybrid_res['max_ms']}ms (Outliers > 5x p95: {hybrid_res['outliers_count']})")

    # Invalidate cache
    requests.post(CACHE_INVALIDATE_URL, timeout=10)

    # 7. Benchmark 3: PRISMX Uncached (100 BENCH queries)
    print("\n>>> 3/5 Benchmarking PRISMX (Hybrid + INT8 Rerank K=10, 100 queries, uncached)...")
    prismx_res = run_sequential_queries(
        bench_queries,
        mode="prismx",
        top_k=5,
        rerank_k=10,
        total_deadline_ms=250.0,
        rerank_budget_ms=200.0,
        use_cache=False,
        scenario_label="prismx_uncached",
    )
    write_csv(prismx_res["records"], OUT_DIR / "raw_latency_prismx_bench100.csv")
    print(f"   PRISMX: p50={prismx_res['p50_ms']}ms | p90={prismx_res['p90_ms']}ms | p95={prismx_res['p95_ms']}ms | p99={prismx_res['p99_ms']}ms | max={prismx_res['max_ms']}ms (Outliers > 5x p95: {prismx_res['outliers_count']})")

    # 8. Benchmark 4: Cache Workload 1: All-Unique Queries (0% hit rate, cache enabled)
    print("\n>>> 4/5 Benchmarking PRISMX Cache: All-Unique Queries (0% hit rate)...")
    requests.post(CACHE_INVALIDATE_URL, timeout=10)
    cache_unique_res = run_sequential_queries(
        bench_queries,
        mode="prismx",
        top_k=5,
        rerank_k=10,
        total_deadline_ms=250.0,
        rerank_budget_ms=200.0,
        use_cache=True,
        scenario_label="cache_all_unique",
    )
    write_csv(cache_unique_res["records"], OUT_DIR / "raw_latency_cache_all_unique.csv")
    print(f"   Cache Unique: p50={cache_unique_res['p50_ms']}ms | p90={cache_unique_res['p90_ms']}ms | p95={cache_unique_res['p95_ms']}ms | p99={cache_unique_res['p99_ms']}ms | max={cache_unique_res['max_ms']}ms")

    # 9. Benchmark 5: Cache Workload 2: 30% Repeated Queries
    print("\n>>> 5/5 Benchmarking PRISMX Cache: 30% Repeated Queries Workload...")
    # FIX: Invalidate cache before starting the 30% repeated workload to prevent carrying over 100% pre-warmed entries from Workload 1
    requests.post(CACHE_INVALIDATE_URL, timeout=10)
    rng = random.Random(42)
    workload_30 = []
    base_pool = bench_queries[:70]
    repeated_pool = bench_queries[:30]
    for _ in range(100):
        if rng.random() < 0.30:
            workload_30.append(rng.choice(repeated_pool))
        else:
            workload_30.append(rng.choice(base_pool))

    cache_30_res = run_sequential_queries(
        workload_30,
        mode="prismx",
        top_k=5,
        rerank_k=10,
        total_deadline_ms=250.0,
        rerank_budget_ms=200.0,
        use_cache=True,
        scenario_label="cache_30pct_repeated",
    )
    write_csv(cache_30_res["records"], OUT_DIR / "raw_latency_cache_30pct_repeated.csv")
    hits_30 = sum(1 for r in cache_30_res["records"] if r["cache_hit"])
    print(f"   Cache 30%:    p50={cache_30_res['p50_ms']}ms | p90={cache_30_res['p90_ms']}ms | p95={cache_30_res['p95_ms']}ms | Hits={hits_30}/100")

    # 10. Assemble and Save Summary JSON
    summary = {
        "benchmark": "Gate 5.5 Official Idle-Machine HTTP Latency Benchmark (c100k_raw)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
        "config_hash": meta["config_hash"],
        "dataset": "c100k_raw",
        "n_bench_queries": 100,
        "machine_state": power_state,
        "environment": env_info,
        "warmup_discarded_20": {k: v for k, v in warmup_res.items() if k != "records"},
        "modes_uncached": {
            "dense": {k: v for k, v in dense_res.items() if k != "records"},
            "hybrid": {k: v for k, v in hybrid_res.items() if k != "records"},
            "prismx": {k: v for k, v in prismx_res.items() if k != "records"},
        },
        "cache_scenarios": {
            "all_unique": {k: v for k, v in cache_unique_res.items() if k != "records"},
            "repeated_30pct": {k: v for k, v in cache_30_res.items() if k != "records"},
        },
        "adr018_rule_evaluation": {
            "condition_i_quality_stat_sig": True,
            "condition_i_detail": "BENCH paired-bootstrap CI of Rerank minus Hybrid on MRR@10 excludes 0 (+0.0583 [+0.0015, +0.1165])",
            "condition_ii_p95_target_ms": 250.0,
            "prismx_uncached_p95_ms": prismx_res["p95_ms"],
            "condition_ii_met": prismx_res["p95_ms"] <= 250.0,
            "mechanical_decision": "prismx" if (prismx_res["p95_ms"] <= 250.0) else "hybrid"
        }
    }

    out_file = OUT_DIR / "latency_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n====================================================================")
    print("LATENCY BENCHMARK COMPLETED")
    print(f"Summary JSON saved to: {out_file}")
    print("====================================================================")


if __name__ == "__main__":
    main()
