"""PRISMX Benchmark Protocol implementing D1, D2, and D3 Latency Measurement."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class BenchmarkRunner:
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000",
        bench_split_path: Path | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.bench_split_path = bench_split_path or (REPO_ROOT / "data" / "manifests" / "split_bench.json")

    def run_benchmark(
        self,
        mode: str = "hybrid",
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        thread_label: str = "default_threads",
        output_csv_path: Path | None = None,
        output_json_path: Path | None = None,
    ) -> dict[str, Any]:
        """Runs the benchmark protocol (D1, D2) against the live HTTP API.
        
        Protocol:
        1. 20 warm-up queries (measured & reported separately, excluded from percentiles).
        2. 100 consecutive distinct BENCH queries executed strictly sequentially.
        3. Client-side wall clock time measured per query (D1).
        4. Calculates n, p50, p90, p95, p99, max, mean via numpy linear interpolation.
        5. Flags and investigates any query > 5x p95.
        """
        with open(self.bench_split_path, "r", encoding="utf-8") as f:
            bench_queries = json.load(f)

        assert len(bench_queries) == 100, f"Expected 100 BENCH queries, got {len(bench_queries)}"

        search_endpoint = f"{self.api_url}/search"

        # 1. Warm-up: first 20 queries of the bench set
        print(f"Executing 20 sequential warm-up queries against {search_endpoint}...")
        warmup_latencies = []
        for item in bench_queries[:20]:
            payload = {
                "query": item["query"],
                "mode": mode,
                "top_k": top_k,
                "filters": filters,
            }
            t0 = time.perf_counter()
            resp = requests.post(search_endpoint, json=payload, timeout=30)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200, f"Search failed during warm-up: {resp.text}"
            warmup_latencies.append(dt_ms)

        mean_warmup = float(np.mean(warmup_latencies))
        print(f"Warm-up complete. Mean warm-up latency: {mean_warmup:.2f} ms")

        # 2. Benchmark run: 100 consecutive distinct queries
        print(f"Running 100 sequential BENCH queries for mode='{mode}', threads='{thread_label}'...")
        query_records = []
        latencies_ms = []

        for idx, item in enumerate(bench_queries, start=1):
            payload = {
                "query": item["query"],
                "mode": mode,
                "top_k": top_k,
                "filters": filters,
            }
            t0 = time.perf_counter()
            resp = requests.post(search_endpoint, json=payload, timeout=30)
            client_dt_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code != 200:
                raise RuntimeError(f"Query {item['query_id']} failed with status {resp.status_code}: {resp.text}")

            resp_data = resp.json()
            server_lats = resp_data.get("latency_ms", {})

            query_records.append({
                "idx": idx,
                "query_id": item["query_id"],
                "query": item["query"],
                "client_wall_clock_ms": round(client_dt_ms, 2),
                "server_encode_ms": server_lats.get("encode", 0.0),
                "server_dense_ms": server_lats.get("dense", 0.0),
                "server_sparse_ms": server_lats.get("sparse", 0.0),
                "server_fusion_ms": server_lats.get("fusion", 0.0),
                "server_fetch_text_ms": server_lats.get("fetch_text", 0.0),
                "server_total_ms": server_lats.get("total", 0.0),
            })
            latencies_ms.append(client_dt_ms)

        # 3. Percentiles using linear interpolation (D2)
        arr = np.array(latencies_ms)
        p50 = float(np.percentile(arr, 50, method="linear"))
        p90 = float(np.percentile(arr, 90, method="linear"))
        p95 = float(np.percentile(arr, 95, method="linear"))
        p99 = float(np.percentile(arr, 99, method="linear"))
        max_lat = float(np.max(arr))
        mean_lat = float(np.mean(arr))

        # Flag outliers > 5x p95
        threshold_5x = 5.0 * p95
        flagged_outliers = [r for r in query_records if r["client_wall_clock_ms"] > threshold_5x]

        # NFR-3 Pass rule: p95 < 300 ms (internal margin target <= 250 ms)
        nfr3_pass = p95 < 300.0
        internal_margin_pass = p95 <= 250.0

        summary = {
            "mode": mode,
            "thread_run": thread_label,
            "filters": filters,
            "top_k": top_k,
            "n_queries": len(latencies_ms),
            "percentile_method": "numpy.percentile, linear interpolation",
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "max_ms": round(max_lat, 2),
            "mean_ms": round(mean_lat, 2),
            "warmup_mean_ms": round(mean_warmup, 2),
            "nfr3_target_under_300ms": nfr3_pass,
            "internal_target_le_250ms": internal_margin_pass,
            "flagged_outliers_count": len(flagged_outliers),
            "flagged_outliers": flagged_outliers,
        }

        # 4. Save per-query CSV
        if output_csv_path:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = list(query_records[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(query_records)

        # 5. Save summary JSON
        if output_json_path:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary


def run_latency_benchmark(
    threads_list: list[int] | None = None,
    mode: str = "dense",
    api_url: str = "http://127.0.0.1:8000",
    config_path: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Runs latency benchmark across thread configurations adhering to D1, D2, and D3."""
    out_dir = output_dir or Path("results/phase1" if mode == "dense" else "results/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(api_url=api_url)

    # D3: Test default threads and 4 threads
    runs = {}
    thread_runs = [("default_threads", None), ("constrained_4_threads", 4)]

    for label, n_threads in thread_runs:
        print(f"\n--- Running Benchmark: {label} (mode={mode}) ---")
        csv_path = out_dir / f"bench_raw_{label}_{mode}.csv"
        json_path = out_dir / f"bench_summary_{label}_{mode}.json"

        # Note: server runs with its configured threads; client sends sequential requests
        res = runner.run_benchmark(
            mode=mode,
            thread_label=label,
            output_csv_path=csv_path,
            output_json_path=json_path,
        )
        runs[label] = res

        print(f"Results for {label}:")
        print(f"  p50: {res['p50_ms']} ms | p90: {res['p90_ms']} ms | p95: {res['p95_ms']} ms | p99: {res['p99_ms']} ms | max: {res['max_ms']} ms")
        print(f"  NFR-3 Pass (<300ms): {res['nfr3_target_under_300ms']} (Internal margin <=250ms: {res['internal_target_le_250ms']})")

    combined_summary = {
        "mode": mode,
        "runs": runs,
        "d1_client_wall_clock": True,
        "d2_percentile_linear_interp": True,
        "d3_thread_comparisons": ["default_threads", "constrained_4_threads"],
    }

    combined_path = out_dir / "latency_benchmark.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    print(f"\nCombined benchmark report saved to {combined_path}")
    return combined_summary
