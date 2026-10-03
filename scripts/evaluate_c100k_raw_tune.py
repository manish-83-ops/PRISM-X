"""Comprehensive evaluation on c100k_raw TUNE split (500 queries) per Gate 5.2 Step 3.
Evaluates:
- Dense baseline
- Hybrid retrieval (alpha=0.8, min-max normalized)
- Hybrid + Rerank (K=10, MiniLM-L6 INT8, max_length=128, 200ms governor)
Computes:
- Hit@1, MRR@10, NDCG@5, NDCG@10, Recall@10, Recall@50
- 95% bootstrap confidence intervals (10,000 resamples)
- Paired bootstrap differences (10,000 resamples) for Hybrid - Dense and Rerank - Hybrid
- Win / Loss / Tie counts
- Informational alpha sensitivity table on {0.6, 0.7, 0.8, 0.9, 1.0} labeled 'not used for selection'
- Breakdown by query_type
- Mean number of gold passages per query
- Fraction of queries with a top-1 that is a non-gold sibling
BENCH split is strictly NOT evaluated.
"""

import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from qdrant_client import models

from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore
from prismx.index.text_store import TextStore
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.rerank import CrossEncoderReranker
from prismx.eval.metrics import hit_at_1, mrr_at_k, ndcg_at_k, recall_at_k

REPO_ROOT = Path(__file__).resolve().parent.parent
TUNE_FILE = REPO_ROOT / "data" / "c100k_raw" / "tune_raw_500.json"
DB_PATH = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
COLLECTION_NAME = "c100k_raw"
OUT_RESULTS = REPO_ROOT / "results" / "c100k_raw" / "tune_eval_results.json"


def paired_bootstrap(diffs: np.ndarray, n_resamples: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    mean_diff = float(np.mean(diffs))
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return mean_diff, low, high


def bootstrap_ci(vals: list[float], n_resamples: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    arr = np.array(vals, dtype=np.float64)
    mean_val = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return mean_val, low, high


def count_win_loss_tie(a_vals: list[float], b_vals: list[float], eps: float = 1e-6) -> dict[str, int]:
    wins = sum(1 for a, b in zip(a_vals, b_vals) if a - b > eps)
    losses = sum(1 for a, b in zip(a_vals, b_vals) if b - a > eps)
    ties = len(a_vals) - wins - losses
    return {"wins": wins, "losses": losses, "ties": ties}


def run_tune_evaluation(chosen_ef: int = 64):
    print("=================================================================")
    print(f"STARTING C100K_RAW TUNE EVALUATION (500 QUERIES, EF={chosen_ef}, QUALITY ONLY)")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"SQLite DB: {DB_PATH}")
    print("=================================================================")

    with open(TUNE_FILE, "r", encoding="utf-8") as f:
        tune_queries = json.load(f)
    n_queries = len(tune_queries)
    print(f"Loaded {n_queries} TUNE queries.")

    # 1. Initialize Components
    print("\nInitializing PRISMX components...")
    qdrant_store = QdrantStore(host="127.0.0.1", port=6333, collection_name=COLLECTION_NAME)
    encoder = DenseEncoder(
        model_name="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
        max_seq_length=128,
        torch_threads=8,
    )
    tokenizer = BM25Tokenizer()
    text_store = TextStore(db_path=str(DB_PATH))
    dense_retriever = DenseRetriever(encoder=encoder, qdrant_store=qdrant_store, default_candidate_depth=50)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        tokenizer=tokenizer,
        qdrant_store=qdrant_store,
        default_candidate_depth=50,
        default_alpha=0.8,
    )
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        torch_threads=8,
    )

    # 2. Evaluation Containers
    modes = ["dense", "hybrid", "hybrid_rerank_k10"]
    metrics_keys = ["hit1", "mrr10", "ndcg5", "ndcg10", "recall10", "recall50"]
    raw_scores: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in metrics_keys} for m in modes
    }

    # Sibling tracking
    top1_is_sibling: dict[str, list[bool]] = {m: [] for m in modes}

    # Query type breakdown
    query_types = sorted(list(set(q["query_type"] for q in tune_queries)))
    qtype_scores: dict[str, dict[str, dict[str, list[float]]]] = {
        qt: {m: {k: [] for k in metrics_keys} for m in modes} for qt in query_types
    }

    # Gold counts
    gold_counts = [len(q["gold_pids"]) for q in tune_queries]
    mean_gold = float(np.mean(gold_counts))

    print(f"Mean number of gold passages per query: {mean_gold:.4f}")

    # 3. Main Evaluation Loop over 500 queries
    print("\nRunning retrieval and reranking across 500 TUNE queries...")
    t_start = time.time()

    for idx, q_item in enumerate(tune_queries):
        qid = q_item["query_id"]
        q_text = q_item["query"]
        q_type = q_item["query_type"]
        gold_ids = set(str(g) for g in q_item["gold_pids"])
        cand_pids = set(str(c) for c in q_item.get("candidate_pids", []))
        non_gold_siblings = cand_pids - gold_ids

        t_req_start = time.perf_counter()
        # --- A. Dense Retrieval (depth=50) ---
        dense_cands, _, _ = dense_retriever.retrieve(query=q_text, limit=50, search_ef=chosen_ef)
        dense_pids = [c["passage_id"] for c in dense_cands]

        # --- B. Hybrid Retrieval (alpha=0.8, depth=50) ---
        fused_cands, _ = hybrid_retriever.retrieve(query=q_text, limit=50, alpha=0.8, norm_method="minmax", search_ef=chosen_ef)
        hybrid_pids = [c["passage_id"] for c in fused_cands]

        # --- C. Hybrid + Rerank (K=10, total_deadline_ms=250, rerank_budget=200ms) ---
        top10_cands = fused_cands[:10]
        hydrated_map = text_store.get_passages_by_ids([c["passage_id"] for c in top10_cands])
        rerank_input = []
        for c in top10_cands:
            c_pid = c["passage_id"]
            c_text = hydrated_map.get(c_pid, {}).get("text", "")
            rerank_input.append({
                "passage_id": c_pid,
                "text": c_text,
                "score": c["score"],
                "fused_rank": c.get("fused_rank")
            })

        elapsed_pre_rerank = (time.perf_counter() - t_req_start) * 1000.0
        remaining_for_rerank = 250.0 - elapsed_pre_rerank - 10.0
        effective_deadline = min(200.0, max(0.0, remaining_for_rerank))

        t_rerank_start = time.perf_counter()
        reranked_top10, _, _ = reranker.rerank(
            query=q_text,
            candidates=rerank_input,
            top_k=10,
            max_length=128,
            deadline_ms=effective_deadline,
            t_request_start=t_rerank_start,
            batch_size=5
        )
        rerank_top10_pids = [c["passage_id"] for c in reranked_top10]
        # Full 50 list: reranked top 10 followed by remaining 40 hybrid
        rerank_50_pids = rerank_top10_pids + hybrid_pids[10:]

        pid_lists = {
            "dense": dense_pids,
            "hybrid": hybrid_pids,
            "hybrid_rerank_k10": rerank_50_pids
        }

        # Calculate metrics for all 3 modes
        for m in modes:
            pids = pid_lists[m]
            h1 = hit_at_1(pids, gold_ids)
            m10 = mrr_at_k(pids, gold_ids, 10)
            n5 = ndcg_at_k(pids, gold_ids, 5)
            n10 = ndcg_at_k(pids, gold_ids, 10)
            r10 = recall_at_k(pids, gold_ids, 10)
            r50 = recall_at_k(pids, gold_ids, 50)

            raw_scores[m]["hit1"].append(h1)
            raw_scores[m]["mrr10"].append(m10)
            raw_scores[m]["ndcg5"].append(n5)
            raw_scores[m]["ndcg10"].append(n10)
            raw_scores[m]["recall10"].append(r10)
            raw_scores[m]["recall50"].append(r50)

            # Query type breakdown
            qtype_scores[q_type][m]["hit1"].append(h1)
            qtype_scores[q_type][m]["mrr10"].append(m10)
            qtype_scores[q_type][m]["ndcg5"].append(n5)
            qtype_scores[q_type][m]["ndcg10"].append(n10)
            qtype_scores[q_type][m]["recall10"].append(r10)
            qtype_scores[q_type][m]["recall50"].append(r50)

            # Sibling check
            top1_pid = pids[0] if pids else None
            top1_is_sibling[m].append(top1_pid in non_gold_siblings)

        if (idx + 1) % 100 == 0 or idx == n_queries - 1:
            elapsed = time.time() - t_start
            print(f"Processed {idx + 1}/{n_queries} queries ({elapsed:.1f}s) | Dense MRR@10: {np.mean(raw_scores['dense']['mrr10']):.4f} | Hybrid MRR@10: {np.mean(raw_scores['hybrid']['mrr10']):.4f} | Rerank MRR@10: {np.mean(raw_scores['hybrid_rerank_k10']['mrr10']):.4f}")

    # 4. Informational Alpha Sensitivity Sweep on Hybrid {0.6, 0.7, 0.8, 0.9, 1.0}
    print("\nRunning Informational Alpha Sensitivity Sweep on Hybrid {0.6, 0.7, 0.8, 0.9, 1.0} (labeled 'not used for selection')...")
    alpha_candidates = [0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_results: dict[str, dict[str, float]] = {}

    for a in alpha_candidates:
        a_h1, a_m10, a_n5, a_n10, a_r10, a_r50 = [], [], [], [], [], []
        for q_item in tune_queries:
            q_text = q_item["query"]
            gold_ids = set(str(g) for g in q_item["gold_pids"])
            fused, _ = hybrid_retriever.retrieve(query=q_text, limit=50, alpha=a, norm_method="minmax", search_ef=chosen_ef)
            pids = [c["passage_id"] for c in fused]
            a_h1.append(hit_at_1(pids, gold_ids))
            a_m10.append(mrr_at_k(pids, gold_ids, 10))
            a_n5.append(ndcg_at_k(pids, gold_ids, 5))
            a_n10.append(ndcg_at_k(pids, gold_ids, 10))
            a_r10.append(recall_at_k(pids, gold_ids, 10))
            a_r50.append(recall_at_k(pids, gold_ids, 50))

        alpha_results[f"alpha_{a:.1f}"] = {
            "alpha": a,
            "hit1": round(float(np.mean(a_h1)), 4),
            "mrr10": round(float(np.mean(a_m10)), 4),
            "ndcg5": round(float(np.mean(a_n5)), 4),
            "ndcg10": round(float(np.mean(a_n10)), 4),
            "recall10": round(float(np.mean(a_r10)), 4),
            "recall50": round(float(np.mean(a_r50)), 4),
        }
        print(f"  Alpha {a:.1f}: Hit@1={alpha_results[f'alpha_{a:.1f}']['hit1']}, MRR@10={alpha_results[f'alpha_{a:.1f}']['mrr10']}, NDCG@5={alpha_results[f'alpha_{a:.1f}']['ndcg5']}, Recall@10={alpha_results[f'alpha_{a:.1f}']['recall10']}")

    # 5. Compute Summary Tables with Bootstrap CIs (10,000 resamples)
    print("\nComputing 10,000 bootstrap resamples for summary metrics...")
    summary_tables: dict[str, Any] = {}
    for m in modes:
        summary_tables[m] = {}
        for k in metrics_keys:
            mean_v, low, high = bootstrap_ci(raw_scores[m][k], n_resamples=10000, seed=42)
            summary_tables[m][k] = {
                "mean": round(mean_v, 4),
                "ci_lower": round(low, 4),
                "ci_upper": round(high, 4),
            }

    # 6. Paired Bootstrap Differences & Win/Loss/Ties (10,000 resamples)
    print("Computing 10,000 paired bootstrap resamples for Hybrid - Dense and Rerank - Hybrid...")
    paired_comparisons: dict[str, Any] = {
        "hybrid_minus_dense": {},
        "rerank_minus_hybrid": {}
    }

    for k in metrics_keys:
        # Hybrid minus Dense
        diff_hd = np.array(raw_scores["hybrid"][k]) - np.array(raw_scores["dense"][k])
        mean_d, low_d, high_d = paired_bootstrap(diff_hd, n_resamples=10000, seed=42)
        wlt_hd = count_win_loss_tie(raw_scores["hybrid"][k], raw_scores["dense"][k])
        paired_comparisons["hybrid_minus_dense"][k] = {
            "mean_delta": round(mean_d, 4),
            "ci_lower": round(low_d, 4),
            "ci_upper": round(high_d, 4),
            "wins": wlt_hd["wins"],
            "losses": wlt_hd["losses"],
            "ties": wlt_hd["ties"],
            "significant_95": bool(low_d > 0.0 or high_d < 0.0)
        }

        # Rerank minus Hybrid
        diff_rh = np.array(raw_scores["hybrid_rerank_k10"][k]) - np.array(raw_scores["hybrid"][k])
        mean_r, low_r, high_r = paired_bootstrap(diff_rh, n_resamples=10000, seed=42)
        wlt_rh = count_win_loss_tie(raw_scores["hybrid_rerank_k10"][k], raw_scores["hybrid"][k])
        paired_comparisons["rerank_minus_hybrid"][k] = {
            "mean_delta": round(mean_r, 4),
            "ci_lower": round(low_r, 4),
            "ci_upper": round(high_r, 4),
            "wins": wlt_rh["wins"],
            "losses": wlt_rh["losses"],
            "ties": wlt_rh["ties"],
            "significant_95": bool(low_r > 0.0 or high_r < 0.0)
        }

    # 7. Query Type Breakdown
    qtype_breakdown: dict[str, Any] = {}
    for qt in query_types:
        count_qt = len(qtype_scores[qt]["dense"]["hit1"])
        pct_qt = count_qt / n_queries * 100.0
        qtype_breakdown[qt] = {
            "count": count_qt,
            "percentage": round(pct_qt, 2),
            "metrics": {
                m: {
                    k: round(float(np.mean(qtype_scores[qt][m][k])), 4)
                    for k in metrics_keys
                } for m in modes
            }
        }

    # 8. Sibling Fraction
    sibling_fractions = {
        m: round(float(np.mean(top1_is_sibling[m])), 4)
        for m in modes
    }

    # Assemble Output Document
    final_output = {
        "benchmark": "c100k_raw TUNE (Quality Evaluation)",
        "dataset": "c100k_raw",
        "split": "tune_raw_500",
        "n_queries": n_queries,
        "mean_gold_passages_per_query": round(mean_gold, 4),
        "fraction_top1_non_gold_sibling": sibling_fractions,
        "frozen_parameters": {
            "dense_model": "BAAI/bge-small-en-v1.5",
            "sparse_engine": "Qdrant BM25 sparse modifier=IDF",
            "fusion": "min-max weighted linear (alpha=0.80)",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2 (INT8 dynamic)",
            "rerank_k": 10,
            "rerank_max_length": 128,
            "rerank_budget_ms": 200.0,
            "total_deadline_ms": 250.0,
            "hnsw_ef": chosen_ef
        },
        "summary_metrics": summary_tables,
        "paired_comparisons": paired_comparisons,
        "alpha_sensitivity_informational": {
            "label": "not used for selection",
            "results": alpha_results
        },
        "query_type_breakdown": qtype_breakdown,
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nEvaluation successfully written to {OUT_RESULTS}!")

    # Print Formatted Markdown Tables
    print("\n=================================================================")
    print("### MAIN TUNE EVALUATION TABLE (500 Queries on c100k_raw)")
    print("=================================================================")
    header = "| Metric | Dense Baseline | Hybrid (alpha=0.8) | Hybrid + Rerank (K=10) | Delta (Hybrid - Dense) [95% CI] (W/L/T) | Delta (Rerank - Hybrid) [95% CI] (W/L/T) |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(sep)
    for k in metrics_keys:
        d_val = f"{summary_tables['dense'][k]['mean']:.4f} [{summary_tables['dense'][k]['ci_lower']:.4f}, {summary_tables['dense'][k]['ci_upper']:.4f}]"
        h_val = f"{summary_tables['hybrid'][k]['mean']:.4f} [{summary_tables['hybrid'][k]['ci_lower']:.4f}, {summary_tables['hybrid'][k]['ci_upper']:.4f}]"
        r_val = f"{summary_tables['hybrid_rerank_k10'][k]['mean']:.4f} [{summary_tables['hybrid_rerank_k10'][k]['ci_lower']:.4f}, {summary_tables['hybrid_rerank_k10'][k]['ci_upper']:.4f}]"

        hd = paired_comparisons["hybrid_minus_dense"][k]
        hd_str = f"{hd['mean_delta']:+.4f} [{hd['ci_lower']:+.4f}, {hd['ci_upper']:+.4f}] ({hd['wins']}/{hd['losses']}/{hd['ties']})"

        rh = paired_comparisons["rerank_minus_hybrid"][k]
        rh_str = f"{rh['mean_delta']:+.4f} [{rh['ci_lower']:+.4f}, {rh['ci_upper']:+.4f}] ({rh['wins']}/{rh['losses']}/{rh['ties']})"

        print(f"| **{k.upper()}** | {d_val} | {h_val} | {r_val} | {hd_str} | {rh_str} |")

    print("\n=================================================================")
    print("### ALPHA SENSITIVITY TABLE (Informational, NOT used for selection)")
    print("=================================================================")
    print("| Alpha | Hit@1 | MRR@10 | NDCG@5 | NDCG@10 | Recall@10 | Recall@50 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for ak, av in alpha_results.items():
        print(f"| **{av['alpha']:.1f}** | {av['hit1']:.4f} | {av['mrr10']:.4f} | {av['ndcg5']:.4f} | {av['ndcg10']:.4f} | {av['recall10']:.4f} | {av['recall50']:.4f} |")

    print("\n=================================================================")
    print("### BREAKDOWN BY QUERY TYPE")
    print("=================================================================")
    for qt, qdata in qtype_breakdown.items():
        print(f"\n#### Query Type: {qt} (N={qdata['count']}, {qdata['percentage']}%)")
        print("| Configuration | Hit@1 | MRR@10 | NDCG@5 | NDCG@10 | Recall@10 | Recall@50 |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for m in modes:
            m_res = qdata["metrics"][m]
            print(f"| **{m}** | {m_res['hit1']:.4f} | {m_res['mrr10']:.4f} | {m_res['ndcg5']:.4f} | {m_res['ndcg10']:.4f} | {m_res['recall10']:.4f} | {m_res['recall50']:.4f} |")

    print("\n=================================================================")
    print("### SIBLING & GOLD STATISTICS")
    print("=================================================================")
    print(f"Mean gold passages per query: {mean_gold:.4f}")
    print("Fraction of queries with top-1 that is a non-gold sibling:")
    for m in modes:
        print(f"  {m}: {sibling_fractions[m]*100:.2f}% ({sibling_fractions[m]:.4f})")
    print("=================================================================")

    return final_output


if __name__ == "__main__":
    import sys
    import yaml
    ef = 64
    if len(sys.argv) > 1:
        ef = int(sys.argv[1])
    else:
        with open("config/CONFIG.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            ef = cfg.get("qdrant", {}).get("search_ef", 64)
    run_tune_evaluation(chosen_ef=ef)
