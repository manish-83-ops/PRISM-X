"""Tune Fusion Methods and Weights on TUNE Split (500 queries) strictly before evaluating BENCH."""

import json
from pathlib import Path
import time
import requests
import numpy as np

from prismx.eval.metrics import mrr_at_k, recall_at_k, ndcg_at_k, hit_at_1
from prismx.eval.bootstrap import bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parent.parent

def evaluate_fusion_variant(queries, search_url, fusion_params):
    mrr_list = []
    r5_list = []
    ndcg5_list = []
    h1_list = []

    for item in queries:
        qid = str(item["query_id"])
        query = item["query"]
        gold_ids = set(str(g) for g in item["gold_passage_ids"])

        payload = {
            "query": query,
            "mode": "hybrid",
            "top_k": 5,
            "fusion": fusion_params,
        }
        resp = requests.post(search_url, json=payload, timeout=20)
        assert resp.status_code == 200, f"Query {qid} failed: {resp.text}"
        data = resp.json()
        pids = [r["passage_id"] for r in data["results"]]

        mrr_list.append(mrr_at_k(pids, gold_ids, 10))
        r5_list.append(recall_at_k(pids, gold_ids, 5))
        ndcg5_list.append(ndcg_at_k(pids, gold_ids, 5))
        h1_list.append(hit_at_1(pids, gold_ids))

    return {
        "mrr10": np.mean(mrr_list),
        "recall5": np.mean(r5_list),
        "ndcg5": np.mean(ndcg5_list),
        "hit1": np.mean(h1_list),
        "raw_mrr": mrr_list,
        "raw_r5": r5_list,
        "raw_ndcg5": ndcg5_list,
    }

def main():
    tune_path = REPO_ROOT / "data" / "manifests" / "split_tune.json"
    with open(tune_path, "r", encoding="utf-8") as f:
        tune_queries = json.load(f)

    # Sample 100 queries from TUNE for fast grid search, or all 500
    eval_set = tune_queries[:150]
    search_url = "http://127.0.0.1:8000/search"

    print("====================================================================")
    print(f"TUNING FUSION STRATEGIES ON TUNE SPLIT (N={len(eval_set)} queries)")
    print("====================================================================")

    variants = [
        {"method": "weighted", "alpha": 0.3, "normalization": "minmax", "label": "Weighted (alpha=0.3, minmax)"},
        {"method": "weighted", "alpha": 0.5, "normalization": "minmax", "label": "Weighted (alpha=0.5, minmax)"},
        {"method": "weighted", "alpha": 0.7, "normalization": "minmax", "label": "Weighted (alpha=0.7, minmax)"},
        {"method": "weighted", "alpha": 0.8, "normalization": "minmax", "label": "Weighted (alpha=0.8, minmax)"},
        {"method": "weighted", "alpha": 0.7, "normalization": "minmax_clipped", "label": "Weighted (alpha=0.7, clipped)"},
        {"method": "rrf", "rrf_k": 20, "label": "RRF (k=20)"},
        {"method": "rrf", "rrf_k": 60, "label": "RRF (k=60)"},
        {"method": "rrf", "rrf_k": 100, "label": "RRF (k=100)"},
    ]

    results_table = []
    best_variant = None
    best_score = -1.0

    for var in variants:
        t0 = time.time()
        f_params = {
            "method": var["method"],
            "alpha": var.get("alpha", 0.7),
            "rrf_k": var.get("rrf_k", 60),
            "normalization": var.get("normalization", "minmax"),
        }
        res = evaluate_fusion_variant(eval_set, search_url, f_params)
        elapsed = time.time() - t0
        row = {
            "label": var["label"],
            "method": var["method"],
            "alpha": var.get("alpha"),
            "rrf_k": var.get("rrf_k"),
            "norm": var.get("normalization", "-"),
            "ndcg5": round(res["ndcg5"], 4),
            "mrr10": round(res["mrr10"], 4),
            "recall5": round(res["recall5"], 4),
            "hit1": round(res["hit1"], 4),
            "time_s": round(elapsed, 1),
        }
        results_table.append(row)
        print(f"[{row['label']:<32}] NDCG@5: {row['ndcg5']:.4f} | MRR@10: {row['mrr10']:.4f} | Recall@5: {row['recall5']:.4f} ({row['time_s']}s)")

        if row["ndcg5"] > best_score:
            best_score = row["ndcg5"]
            best_variant = row

    out_file = REPO_ROOT / "results" / "phase2" / "fusion_tuning_tune.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": results_table, "best": best_variant}, f, indent=2)

    print("\n" + "=" * 70)
    print("FUSION COMPARISON TABLE (TUNE SPLIT)")
    print("=" * 70)
    print(f"{'Strategy / Variant':<35} {'NDCG@5':<10} {'MRR@10':<10} {'Recall@5':<10}")
    print("-" * 70)
    for r in results_table:
        print(f"{r['label']:<35} {r['ndcg5']:<10.4f} {r['mrr10']:<10.4f} {r['recall5']:<10.4f}")
    print("=" * 70)
    print(f"\nOptimal Frozen Strategy Selected from TUNE: {best_variant['label']}")

if __name__ == "__main__":
    main()
