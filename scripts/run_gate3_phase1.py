"""Gate 3 Sub-step 3a: Phase 1 Naive Dense Baseline Evaluation on 100 BENCH Queries."""

import json
from pathlib import Path
import time
import requests
import numpy as np

from prismx.eval.metrics import (
    hit_at_1,
    success_at_k,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    dup_aware_hit_at_1,
    dup_aware_recall_at_5,
)
from prismx.eval.bootstrap import bootstrap_ci
from prismx.eval.ragas_eval import compute_non_llm_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    bench_file = REPO_ROOT / "data" / "manifests" / "split_bench.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    assert len(bench_queries) == 100, f"Expected 100 BENCH queries, got {len(bench_queries)}"

    near_dups_file = REPO_ROOT / "data" / "manifests" / "near_duplicates_manifest.json"
    near_dups_map = {}
    if near_dups_file.exists():
        with open(near_dups_file, "r", encoding="utf-8") as f:
            near_dups_map = json.load(f).get("near_duplicates_by_gold_id", {})

    search_url = "http://127.0.0.1:8000/search"
    results_dir = REPO_ROOT / "results" / "phase1"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================================")
    print("GATE 3a: RUNNING PHASE 1 DENSE BASELINE ON 100 BENCH QUERIES")
    print("====================================================================")

    # 1. Warm-up
    print("Executing 10 warm-up queries (excluded from metrics)...")
    for item in bench_queries[:10]:
        requests.post(search_url, json={"query": item["query"], "mode": "dense", "top_k": 5}, timeout=10)

    # 2. Main run over 100 BENCH queries
    query_records = []
    latencies = []
    per_query_metrics = {
        "hit_at_1": [],
        "success_at_5": [],
        "recall_at_5": [],
        "recall_at_10": [],
        "recall_at_20": [],
        "mrr_at_10": [],
        "ndcg_at_5": [],
        "ndcg_at_10": [],
        "dup_aware_hit_at_1": [],
        "dup_aware_recall_at_5": [],
        "ragas_context_precision": [],
        "ragas_context_recall": [],
    }

    t_start_all = time.time()
    for idx, item in enumerate(bench_queries, start=1):
        qid = str(item["query_id"])
        query = item["query"]
        gold_ids = set(str(g) for g in item["gold_passage_ids"])

        t0 = time.perf_counter()
        resp = requests.post(
            search_url,
            json={"query": query, "mode": "dense", "top_k": 5},
            timeout=30,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        assert resp.status_code == 200, f"Query failed: {resp.text}"
        data = resp.json()
        retrieved_ids = [r["passage_id"] for r in data["results"]]
        retrieved_texts = [r["text"] for r in data["results"]]

        # Calculate IR metrics
        h1 = hit_at_1(retrieved_ids, gold_ids)
        s5 = success_at_k(retrieved_ids, gold_ids, 5)
        r5 = recall_at_k(retrieved_ids, gold_ids, 5)
        r10 = recall_at_k(retrieved_ids, gold_ids, 10)
        r20 = recall_at_k(retrieved_ids, gold_ids, 20)
        mrr10 = mrr_at_k(retrieved_ids, gold_ids, 10)
        ndcg5 = ndcg_at_k(retrieved_ids, gold_ids, 5)
        ndcg10 = ndcg_at_k(retrieved_ids, gold_ids, 10)
        dh1 = dup_aware_hit_at_1(retrieved_ids, gold_ids, near_dups_map)
        dr5 = dup_aware_recall_at_5(retrieved_ids, gold_ids, near_dups_map)

        # RAGAS metrics (Family A deterministic)
        ragas_m = compute_non_llm_metrics(retrieved_texts, retrieved_ids, gold_ids)

        per_query_metrics["hit_at_1"].append(h1)
        per_query_metrics["success_at_5"].append(s5)
        per_query_metrics["recall_at_5"].append(r5)
        per_query_metrics["recall_at_10"].append(r10)
        per_query_metrics["recall_at_20"].append(r20)
        per_query_metrics["mrr_at_10"].append(mrr10)
        per_query_metrics["ndcg_at_5"].append(ndcg5)
        per_query_metrics["ndcg_at_10"].append(ndcg10)
        per_query_metrics["dup_aware_hit_at_1"].append(dh1)
        per_query_metrics["dup_aware_recall_at_5"].append(dr5)
        per_query_metrics["ragas_context_precision"].append(ragas_m["context_precision"])
        per_query_metrics["ragas_context_recall"].append(ragas_m["context_recall"])

        query_records.append({
            "idx": idx,
            "query_id": qid,
            "query": query,
            "latency_ms": round(dt_ms, 2),
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": [round(r["score"], 4) for r in data["results"]],
            "gold_passage_ids": list(gold_ids),
            "hit_at_1": h1,
            "mrr_at_10": round(mrr10, 4),
            "recall_at_5": round(r5, 4),
            "ragas_cp": ragas_m["context_precision"],
            "ragas_cr": ragas_m["context_recall"],
        })

    total_time = time.time() - t_start_all
    print(f"Executed 100 queries in {total_time:.2f}s ({100/total_time:.1f} QPS)")

    # Compute percentiles and summary metrics with 10k bootstrap CIs
    summary_metrics = {}
    for m_name, vals in per_query_metrics.items():
        summary_metrics[m_name] = bootstrap_ci(vals, n_resamples=10000, seed=42)

    lat_arr = np.array(latencies)
    latency_summary = {
        "p50_ms": round(float(np.percentile(lat_arr, 50, method="linear")), 2),
        "p90_ms": round(float(np.percentile(lat_arr, 90, method="linear")), 2),
        "p95_ms": round(float(np.percentile(lat_arr, 95, method="linear")), 2),
        "p99_ms": round(float(np.percentile(lat_arr, 99, method="linear")), 2),
        "max_ms": round(float(np.max(lat_arr)), 2),
        "mean_ms": round(float(np.mean(lat_arr)), 2),
    }

    final_report = {
        "phase": 1,
        "mode": "dense",
        "eval_set": "BENCH (100 queries)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_summary": latency_summary,
        "metrics": summary_metrics,
    }

    # Save outputs
    with open(results_dir / "bench_dense_retrievals.json", "w", encoding="utf-8") as f:
        json.dump(query_records, f, indent=2)

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 1 DENSE BENCHMARK RESULTS (100 BENCH QUERIES)")
    print("=" * 70)
    print(f"Latency: p50={latency_summary['p50_ms']}ms | p90={latency_summary['p90_ms']}ms | p95={latency_summary['p95_ms']}ms | p99={latency_summary['p99_ms']}ms")
    print(f"Target p95 < 300ms: {'PASS' if latency_summary['p95_ms'] < 300 else 'FAIL'}")
    print("-" * 70)
    print(f"{'Metric':<25} {'Mean':<10} {'95% CI Lower':<15} {'95% CI Upper':<15}")
    print("-" * 70)
    for m_name in ["hit_at_1", "mrr_at_10", "recall_at_5", "recall_at_10", "recall_at_20", "ndcg_at_5", "ragas_context_precision", "ragas_context_recall"]:
        ci = summary_metrics[m_name]
        print(f"{m_name:<25} {ci['mean']:<10.4f} {ci['ci_lower']:<15.4f} {ci['ci_upper']:<15.4f}")
    print("=" * 70 + "\n")
    print(f"Saved results to {results_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
