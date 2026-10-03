"""Single-Run BENCH Evaluation on c100k_raw (100 queries) per Gate 5.3 Step 3.
Evaluates:
- Dense baseline
- Hybrid retrieval (alpha=0.8, min-max normalized)
- Hybrid + Rerank (K=10, total_deadline_ms=250, rerank_budget_ms=200)
Computes:
- Hit@1, MRR@10, NDCG@5, NDCG@10, Recall@10, Recall@50 with 95% bootstrap CIs (10,000 resamples)
- Paired bootstrap differences (10,000 resamples) for Hybrid - Dense and Rerank - Hybrid
- Win / Loss / Tie counts
- Governor truncation rate and mean number of candidates actually scored
- Fraction of queries with top-1 that is a non-gold sibling
Logs entry in results/BENCH_USAGE.md.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from qdrant_client import QdrantClient, models

from prismx.config import load_config
from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore
from prismx.index.text_store import TextStore
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.rerank import CrossEncoderReranker
from prismx.eval.metrics import hit_at_1, mrr_at_k, ndcg_at_k, recall_at_k

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_FILE = REPO_ROOT / "data" / "c100k_raw" / "bench_raw_100.json"
DB_PATH = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
COLLECTION_NAME = "c100k_raw"
OUT_RESULTS = REPO_ROOT / "results" / "c100k_raw" / "bench_eval_results.json"
BENCH_USAGE_FILE = REPO_ROOT / "results" / "BENCH_USAGE.md"


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


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


def run_bench_evaluation(chosen_ef: int | None = None):
    cfg = load_config()
    config_hash = cfg.get("_config_hash", "unknown")
    if chosen_ef is None:
        chosen_ef = cfg.get("qdrant", {}).get("search_ef", 64)

    print("=================================================================")
    print("STARTING SINGLE-RUN C100K_RAW BENCH EVALUATION (100 QUERIES)")
    print(f"Config Hash: {config_hash}")
    print(f"HNSW search_ef: {chosen_ef}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"SQLite DB: {DB_PATH}")
    print("=================================================================")

    with open(BENCH_FILE, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)
    n_queries = len(bench_queries)
    print(f"Loaded {n_queries} BENCH queries.")

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

    modes = ["dense", "hybrid", "hybrid_rerank_k10"]
    metrics_keys = ["hit1", "mrr10", "ndcg5", "ndcg10", "recall10", "recall50"]
    raw_scores: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in metrics_keys} for m in modes
    }

    # Governor and sibling metrics
    truncation_events = 0
    exhausted_events = 0
    candidates_scored_counts = []
    top1_is_sibling: dict[str, list[bool]] = {m: [] for m in modes}

    # Query type breakdown
    query_types = sorted(list(set(q["query_type"] for q in bench_queries)))
    qtype_scores: dict[str, dict[str, dict[str, list[float]]]] = {
        qt: {m: {k: [] for k in metrics_keys} for m in modes} for qt in query_types
    }

    gold_counts = [len(q["gold_pids"]) for q in bench_queries]
    mean_gold = float(np.mean(gold_counts))

    print(f"Mean gold passages per BENCH query: {mean_gold:.4f}")
    t_start = time.time()

    for idx, q_item in enumerate(bench_queries):
        qid = q_item["query_id"]
        q_text = q_item["query"]
        gold_ids = set(str(g) for g in q_item["gold_pids"])
        cand_pids = set(str(c) for c in q_item.get("candidate_pids", []))
        non_gold_siblings = cand_pids - gold_ids

        # --- A. Dense Retrieval ---
        # Query with chosen search_ef
        q_emb = encoder.encode_queries(q_text)[0]
        res_dense = qdrant_store.client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb.tolist(),
            using="dense",
            search_params=models.SearchParams(hnsw_ef=chosen_ef),
            limit=50,
            with_payload=True
        )
        dense_pids = [pt.payload["passage_id"] for pt in res_dense.points]

        # --- B. Hybrid Retrieval (alpha=0.8, depth=50) ---
        t_req_start = time.perf_counter()
        fused_cands, _ = hybrid_retriever.retrieve(query=q_text, limit=50, alpha=0.8, norm_method="minmax", search_ef=chosen_ef)
        hybrid_pids = [c["passage_id"] for c in fused_cands]

        # --- C. Hybrid + Rerank (K=10, total_deadline_ms=250.0, rerank_budget_ms=200.0) ---
        t_pre_rerank_elapsed = (time.perf_counter() - t_req_start) * 1000.0
        remaining_for_rerank = 250.0 - t_pre_rerank_elapsed - 10.0
        effective_deadline = min(200.0, max(0.0, remaining_for_rerank))

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

        t_rerank_start = time.perf_counter()
        reranked_top10, _, gov_state = reranker.rerank(
            query=q_text,
            candidates=rerank_input,
            top_k=10,
            max_length=128,
            deadline_ms=effective_deadline,
            t_request_start=t_rerank_start,
            batch_size=5
        )

        n_scored = sum(1 for c in reranked_top10 if c.get("rerank_score") is not None)
        candidates_scored_counts.append(n_scored)

        if gov_state == "truncated":
            truncation_events += 1
        elif gov_state == "exhausted_before_first_batch":
            exhausted_events += 1

        rerank_top10_pids = [c["passage_id"] for c in reranked_top10]
        rerank_50_pids = rerank_top10_pids + hybrid_pids[10:]

        pid_lists = {
            "dense": dense_pids,
            "hybrid": hybrid_pids,
            "hybrid_rerank_k10": rerank_50_pids
        }

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
            qtype_scores[q_item["query_type"]][m]["hit1"].append(h1)
            qtype_scores[q_item["query_type"]][m]["mrr10"].append(m10)
            qtype_scores[q_item["query_type"]][m]["ndcg5"].append(n5)
            qtype_scores[q_item["query_type"]][m]["ndcg10"].append(n10)
            qtype_scores[q_item["query_type"]][m]["recall10"].append(r10)
            qtype_scores[q_item["query_type"]][m]["recall50"].append(r50)

            top1_pid = pids[0] if pids else None
            top1_is_sibling[m].append(top1_pid in non_gold_siblings)

        if (idx + 1) % 25 == 0 or idx == n_queries - 1:
            print(f"  BENCH progress: {idx + 1}/{n_queries} ({time.time() - t_start:.1f}s)")

    # 2. Compute Summary Metrics with 10,000 resamples
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

    # 3. Paired Differences & Win/Loss/Ties (10,000 resamples)
    paired_comparisons: dict[str, Any] = {
        "hybrid_minus_dense": {},
        "rerank_minus_hybrid": {},
        "rerank_minus_dense": {}
    }

    for k in metrics_keys:
        # Hybrid minus Dense
        diff_hd = np.array(raw_scores["hybrid"][k]) - np.array(raw_scores["dense"][k])
        mean_d, low_d, high_d = paired_bootstrap(diff_hd, n_resamples=10000, seed=42)
        wlt_hd = count_win_loss_tie(raw_scores["hybrid"][k], raw_scores["dense"][k])
        # Wording rule: only call something an improvement if CI excludes 0; otherwise "directional, not significant"
        excludes_zero_hd = bool(low_d > 0.0 or high_d < 0.0)
        paired_comparisons["hybrid_minus_dense"][k] = {
            "mean_delta": round(mean_d, 4),
            "ci_lower": round(low_d, 4),
            "ci_upper": round(high_d, 4),
            "wins": wlt_hd["wins"],
            "losses": wlt_hd["losses"],
            "ties": wlt_hd["ties"],
            "ci_excludes_zero": excludes_zero_hd,
            "verdict": "Statistically Significant Improvement" if excludes_zero_hd and mean_d > 0 else ("Statistically Significant Degradation" if excludes_zero_hd and mean_d < 0 else "Directional, not significant (crosses zero)")
        }

        # Rerank minus Hybrid
        diff_rh = np.array(raw_scores["hybrid_rerank_k10"][k]) - np.array(raw_scores["hybrid"][k])
        mean_r, low_r, high_r = paired_bootstrap(diff_rh, n_resamples=10000, seed=42)
        wlt_rh = count_win_loss_tie(raw_scores["hybrid_rerank_k10"][k], raw_scores["hybrid"][k])
        excludes_zero_rh = bool(low_r > 0.0 or high_r < 0.0)
        paired_comparisons["rerank_minus_hybrid"][k] = {
            "mean_delta": round(mean_r, 4),
            "ci_lower": round(low_r, 4),
            "ci_upper": round(high_r, 4),
            "wins": wlt_rh["wins"],
            "losses": wlt_rh["losses"],
            "ties": wlt_rh["ties"],
            "ci_excludes_zero": excludes_zero_rh,
            "verdict": "Statistically Significant Improvement" if excludes_zero_rh and mean_r > 0 else ("Statistically Significant Degradation" if excludes_zero_rh and mean_r < 0 else "Directional, not significant (crosses zero)")
        }

        # Rerank minus Dense
        diff_rd = np.array(raw_scores["hybrid_rerank_k10"][k]) - np.array(raw_scores["dense"][k])
        mean_rd, low_rd, high_rd = paired_bootstrap(diff_rd, n_resamples=10000, seed=42)
        wlt_rd = count_win_loss_tie(raw_scores["hybrid_rerank_k10"][k], raw_scores["dense"][k])
        excludes_zero_rd = bool(low_rd > 0.0 or high_rd < 0.0)
        paired_comparisons["rerank_minus_dense"][k] = {
            "mean_delta": round(mean_rd, 4),
            "ci_lower": round(low_rd, 4),
            "ci_upper": round(high_rd, 4),
            "wins": wlt_rd["wins"],
            "losses": wlt_rd["losses"],
            "ties": wlt_rd["ties"],
            "ci_excludes_zero": excludes_zero_rd,
            "verdict": "Statistically Significant Improvement" if excludes_zero_rd and mean_rd > 0 else ("Statistically Significant Degradation" if excludes_zero_rd and mean_rd < 0 else "Directional, not significant (crosses zero)")
        }

    # Governor and sibling metrics
    truncation_rate = truncation_events / n_queries
    exhausted_rate = exhausted_events / n_queries
    mean_scored = float(np.mean(candidates_scored_counts))
    sibling_fractions = {m: round(float(np.mean(top1_is_sibling[m])), 4) for m in modes}

    commit_hash = get_git_commit()
    date_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Query type breakdown table
    qtype_breakdown = {}
    for qt in query_types:
        q_count = len(qtype_scores[qt]["dense"]["hit1"])
        if q_count == 0:
            continue
        qtype_breakdown[qt] = {
            "count": q_count,
            "percentage": round(q_count / n_queries * 100, 1),
            "metrics": {
                m: {k: round(float(np.mean(qtype_scores[qt][m][k])), 4) for k in metrics_keys}
                for m in modes
            }
        }

    bench_output = {
        "benchmark": "c100k_raw BENCH (Single-Run Evaluation)",
        "dataset": "c100k_raw",
        "split": "bench_raw_100",
        "n_queries": n_queries,
        "date": date_str,
        "commit": commit_hash,
        "single_run_declaration": "This BENCH set is new and has been scored exactly once. BENCH queries were strictly never used for hyperparameter tuning or selection.",
        "configuration": {
            "config_hash": config_hash,
            "hnsw_ef": chosen_ef,
            "alpha": 0.8,
            "rerank_k": 10,
            "total_deadline_ms": 250.0,
            "rerank_budget_ms": 200.0,
            "governor_batch_size": 5
        },
        "governor_telemetry": {
            "truncation_events": truncation_events,
            "truncation_rate": round(truncation_rate, 4),
            "exhausted_before_first_batch_events": exhausted_events,
            "exhausted_rate": round(exhausted_rate, 4),
            "mean_candidates_scored": round(mean_scored, 2)
        },
        "sibling_analysis": {
            "mean_gold_passages_per_query": round(mean_gold, 4),
            "fraction_top1_non_gold_sibling": sibling_fractions
        },
        "summary_metrics": summary_tables,
        "paired_comparisons": paired_comparisons,
        "query_type_breakdown": qtype_breakdown
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(bench_output, f, indent=2)

    # 4. Log in results/BENCH_USAGE.md
    bench_log_entry = f"""
## Run: c100k_raw BENCH Single-Run Evaluation (Gate 5.4)
- **Date:** {date_str}
- **Git Commit:** `{commit_hash}`
- **Config Hash:** `{config_hash}`
- **Corpus:** `c100k_raw` (100,008 unique passages, uncurated candidate distribution)
- **Split:** `data/c100k_raw/bench_raw_100.json` (N=100)
- **Declaration:** This BENCH set is brand new and has been scored **exactly once**. Zero prior evaluations were executed against this split.
- **Serving Configuration:** HNSW `search_ef={chosen_ef}`, Fusion $\\alpha=0.80$ (min-max, not worse than alternatives), Reranker MiniLM-L-6 INT8 ($K=10$, `total_deadline_ms=250.0`, `rerank_budget_ms=200.0`).
- **Governor Telemetry:** Truncation rate = {truncation_rate*100:.1f}%, Exhausted rate = {exhausted_rate*100:.1f}%, Mean candidates scored = {mean_scored:.2f} / 10.
- **Top-1 Non-Gold Sibling Fraction (Unjudged):** Dense = {sibling_fractions['dense']*100:.1f}%, Hybrid = {sibling_fractions['hybrid']*100:.1f}%, Hybrid+Rerank = {sibling_fractions['hybrid_rerank_k10']*100:.1f}%.
- **Summary Metrics (Mean [95% Bootstrap CI]):**
  - Dense Baseline: Hit@1 = {summary_tables['dense']['hit1']['mean']:.4f}, MRR@10 = {summary_tables['dense']['mrr10']['mean']:.4f}, NDCG@5 = {summary_tables['dense']['ndcg5']['mean']:.4f}, NDCG@10 = {summary_tables['dense']['ndcg10']['mean']:.4f}, Recall@10 = {summary_tables['dense']['recall10']['mean']:.4f}, Recall@50 = {summary_tables['dense']['recall50']['mean']:.4f}
  - Hybrid Retrieval: Hit@1 = {summary_tables['hybrid']['hit1']['mean']:.4f}, MRR@10 = {summary_tables['hybrid']['mrr10']['mean']:.4f}, NDCG@5 = {summary_tables['hybrid']['ndcg5']['mean']:.4f}, NDCG@10 = {summary_tables['hybrid']['ndcg10']['mean']:.4f}, Recall@10 = {summary_tables['hybrid']['recall10']['mean']:.4f}, Recall@50 = {summary_tables['hybrid']['recall50']['mean']:.4f}
  - Hybrid + Rerank: Hit@1 = {summary_tables['hybrid_rerank_k10']['hit1']['mean']:.4f}, MRR@10 = {summary_tables['hybrid_rerank_k10']['mrr10']['mean']:.4f}, NDCG@5 = {summary_tables['hybrid_rerank_k10']['ndcg5']['mean']:.4f}, NDCG@10 = {summary_tables['hybrid_rerank_k10']['ndcg10']['mean']:.4f}, Recall@10 = {summary_tables['hybrid_rerank_k10']['recall10']['mean']:.4f}, Recall@50 = {summary_tables['hybrid_rerank_k10']['recall50']['mean']:.4f}
"""
    with open(BENCH_USAGE_FILE, "a", encoding="utf-8") as f:
        f.write(bench_log_entry)

    print("\n=================================================================")
    print("### MAIN BENCH EVALUATION TABLE (100 Queries on c100k_raw)")
    print("=================================================================")
    header = "| Metric | Dense Baseline | Hybrid (alpha=0.8) | Hybrid + Rerank (K=10) | Delta (Hybrid - Dense) [95% CI] (W/L/T) | Delta (Rerank - Hybrid) [95% CI] (W/L/T) | Delta (Rerank - Dense) [95% CI] (W/L/T) |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
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

        rd = paired_comparisons["rerank_minus_dense"][k]
        rd_str = f"{rd['mean_delta']:+.4f} [{rd['ci_lower']:+.4f}, {rd['ci_upper']:+.4f}] ({rd['wins']}/{rd['losses']}/{rd['ties']})"

        print(f"| **{k.upper()}** | {d_val} | {h_val} | {r_val} | {hd_str} | {rh_str} | {rd_str} |")

    print("\n=================================================================")
    print("### BREAKDOWN BY QUERY TYPE (BENCH 100 Queries)")
    print("=================================================================")
    for qt, qdata in qtype_breakdown.items():
        print(f"\n#### Query Type: {qt} (N={qdata['count']}, {qdata['percentage']}%)")
        print("| Configuration | Hit@1 | MRR@10 | NDCG@5 | NDCG@10 | Recall@10 | Recall@50 |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for m in modes:
            m_res = qdata["metrics"][m]
            print(f"| **{m}** | {m_res['hit1']:.4f} | {m_res['mrr10']:.4f} | {m_res['ndcg5']:.4f} | {m_res['ndcg10']:.4f} | {m_res['recall10']:.4f} | {m_res['recall50']:.4f} |")

    print("\n=================================================================")
    print("### GOVERNOR & SIBLING TELEMETRY")
    print("=================================================================")
    print(f"Governor Truncation Rate: {truncation_rate*100:.2f}% ({truncation_events}/{n_queries})")
    print(f"Governor Exhausted-Before-Batch-0 Rate: {exhausted_rate*100:.2f}% ({exhausted_events}/{n_queries})")
    print(f"Mean Candidates Scored by Cross-Encoder: {mean_scored:.2f} / 10")
    print("Fraction of queries with top-1 non-gold sibling:")
    for m in modes:
        print(f"  {m}: {sibling_fractions[m]*100:.2f}% ({sibling_fractions[m]:.4f})")
    print("=================================================================")

    return bench_output


if __name__ == "__main__":
    run_bench_evaluation()
