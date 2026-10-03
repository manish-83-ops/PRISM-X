"""PRISMX Zero-Token Analysis per Gate 12 Part 2.2 and ADR-023.
Computes:
1. Top-1 identity rate and Top-5 Jaccard for Dense/Hybrid/Rerank on TUNE (500) and BENCH (100).
2. Lexical-contribution audit: sparse-only top-1 vs dense top-1, gold distributions, and oracle best-channel bound.
3. CP headroom: fraction of queries with CP < 1, and count/fraction where first useful passage is below rank 1.
4. Top 10 largest Rerank-vs-Hybrid CP drops with passages, ranks, and verdicts.
5. Bootstrap SE and required sample size N for +/-0.02 CI on paired differences.
Zero Groq calls, zero LLM tokens.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prismx.index.text_store import TextStore

# Paths
DATA_DIR = REPO_ROOT / "data" / "c100k_raw"
RESULTS_DIR = REPO_ROOT / "results"
RAGAS_DIR = RESULTS_DIR / "ragas" / "c100k_raw"
DB_PATH = DATA_DIR / "text_store_raw.db"

TUNE_RETRIEVALS_PATH = RESULTS_DIR / "tune_per_query_retrievals.json"
BENCH_C100K_PATH = RESULTS_DIR / "c100k_raw" / "bench_retrievals_c100k_raw.json"
PER_QUERY_SCORES_PATH = RAGAS_DIR / "per_query_scores.csv"
FROZEN_50_PATH = REPO_ROOT / "data" / "manifests" / "frozen_ragas_bench_raw_50.json"
OUT_JSON = RESULTS_DIR / "ragas" / "zero_token_audit.json"


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a.union(set_b)
    if not union:
        return 1.0
    return len(set_a.intersection(set_b)) / len(union)


def compute_overlap_metrics(records_a: list[list[str]], records_b: list[list[str]]) -> dict[str, float]:
    """Computes top-1 identity rate and top-5 mean Jaccard between two retrieval lists."""
    assert len(records_a) == len(records_b)
    top1_identical = 0
    jaccards = []
    for a, b in zip(records_a, records_b):
        if a and b and a[0] == b[0]:
            top1_identical += 1
        jaccards.append(jaccard(set(a[:5]), set(b[:5])))
    return {
        "top1_identity_rate": round(top1_identical / len(records_a), 4),
        "top1_differ_rate": round(1.0 - (top1_identical / len(records_a)), 4),
        "mean_top5_jaccard": round(float(np.mean(jaccards)), 4),
    }


def find_verdicts_for_cp(cp: float) -> list[int]:
    """Reconstructs the 5-element binary verdict vector corresponding to a Ragas CP score.
    There are only 32 possible combinations for top-5.
    """
    best_pattern = [0, 0, 0, 0, 0]
    best_diff = 999.0
    for v0 in [0, 1]:
        for v1 in [0, 1]:
            for v2 in [0, 1]:
                for v3 in [0, 1]:
                    for v4 in [0, 1]:
                        pat = [v0, v1, v2, v3, v4]
                        hits = sum(pat)
                        if hits == 0:
                            score = 0.0
                        else:
                            cum_prec = sum((sum(pat[:i+1]) / (i+1)) * pat[i] for i in range(5))
                            score = cum_prec / hits
                        diff = abs(score - cp)
                        if diff < best_diff:
                            best_diff = diff
                            best_pattern = pat
    return best_pattern


def main():
    print("=" * 70)
    print("STARTING GATE 12 PART 2.2: ZERO-TOKEN RETRIEVAL & RAGAS AUDIT")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Top-1 Identity & Top-5 Jaccard on BENCH (100)
    # -------------------------------------------------------------
    print("\n--- 1. BENCH (100 Queries on c100k_raw) Retrieval Overlap ---")
    with open(BENCH_C100K_PATH, "r", encoding="utf-8") as f:
        bench_c100k = json.load(f)

    bench_dense_pids = [q["dense_pids"] for q in bench_c100k]
    bench_hybrid_pids = [q["hybrid_pids"] for q in bench_c100k]
    bench_rerank_pids = [q["rerank_pids"] for q in bench_c100k]

    bench_hd = compute_overlap_metrics(bench_dense_pids, bench_hybrid_pids)
    bench_hr = compute_overlap_metrics(bench_hybrid_pids, bench_rerank_pids)
    bench_dr = compute_overlap_metrics(bench_dense_pids, bench_rerank_pids)

    print(f"BENCH Hybrid vs Dense : Top-1 Identity = {bench_hd['top1_identity_rate']:.4f}, Top-5 Jaccard = {bench_hd['mean_top5_jaccard']:.4f}")
    print(f"BENCH Rerank vs Hybrid: Top-1 Identity = {bench_hr['top1_identity_rate']:.4f}, Top-5 Jaccard = {bench_hr['mean_top5_jaccard']:.4f}")
    print(f"BENCH Rerank vs Dense : Top-1 Identity = {bench_dr['top1_identity_rate']:.4f}, Top-5 Jaccard = {bench_dr['mean_top5_jaccard']:.4f}")

    # -------------------------------------------------------------
    # 2. Top-1 Identity & Top-5 Jaccard on TUNE (500)
    # -------------------------------------------------------------
    print("\n--- 2. TUNE (500 Queries) Retrieval Overlap ---")
    tune_hd, tune_hr, tune_dr = {}, {}, {}
    tune_data = []
    if TUNE_RETRIEVALS_PATH.exists():
        with open(TUNE_RETRIEVALS_PATH, "r", encoding="utf-8") as f:
            tune_data = json.load(f)
        tune_dense_pids = [q["dense_pids"] for q in tune_data]
        tune_hybrid_pids = [q["hybrid_pids"] for q in tune_data]
        tune_rerank_pids = [q["rerank_pids"] for q in tune_data]

        tune_hd = compute_overlap_metrics(tune_dense_pids, tune_hybrid_pids)
        tune_hr = compute_overlap_metrics(tune_hybrid_pids, tune_rerank_pids)
        tune_dr = compute_overlap_metrics(tune_dense_pids, tune_rerank_pids)

        print(f"TUNE Hybrid vs Dense : Top-1 Identity = {tune_hd['top1_identity_rate']:.4f}, Top-5 Jaccard = {tune_hd['mean_top5_jaccard']:.4f}")
        print(f"TUNE Rerank vs Hybrid: Top-1 Identity = {tune_hr['top1_identity_rate']:.4f}, Top-5 Jaccard = {tune_hr['mean_top5_jaccard']:.4f}")
        print(f"TUNE Rerank vs Dense : Top-1 Identity = {tune_dr['top1_identity_rate']:.4f}, Top-5 Jaccard = {tune_dr['mean_top5_jaccard']:.4f}")
    else:
        print(f"Waiting for {TUNE_RETRIEVALS_PATH}...")

    # -------------------------------------------------------------
    # 3. Lexical-Contribution Audit & Oracle Upper Bound
    # -------------------------------------------------------------
    print("\n--- 3. Lexical-Contribution Audit ---")
    lexical_audit = {}
    if tune_data:
        n_tune = len(tune_data)
        differ_top1 = 0
        sparse_only_top1_gold = 0
        dense_only_top1_gold = 0
        both_top1_gold = 0
        neither_top1_gold = 0

        dense_hit1_count = 0
        sparse_hit1_count = 0
        hybrid_hit1_count = 0
        oracle_hit1_count = 0

        dense_hit5_count = 0
        sparse_hit5_count = 0
        hybrid_hit5_count = 0
        oracle_hit5_count = 0

        for q in tune_data:
            golds = set(q["gold_pids"])
            d_pids = q["dense_pids"]
            s_pids = q["sparse_pids"]
            h_pids = q["hybrid_pids"]

            d_top1 = d_pids[0] if d_pids else None
            s_top1 = s_pids[0] if s_pids else None
            h_top1 = h_pids[0] if h_pids else None

            d_hit1 = d_top1 in golds
            s_hit1 = s_top1 in golds
            h_hit1 = h_top1 in golds
            oracle_hit1 = d_hit1 or s_hit1

            d_hit5 = bool(set(d_pids[:5]).intersection(golds))
            s_hit5 = bool(set(s_pids[:5]).intersection(golds))
            h_hit5 = bool(set(h_pids[:5]).intersection(golds))
            oracle_hit5 = d_hit5 or s_hit5

            if d_hit1: dense_hit1_count += 1
            if s_hit1: sparse_hit1_count += 1
            if h_hit1: hybrid_hit1_count += 1
            if oracle_hit1: oracle_hit1_count += 1

            if d_hit5: dense_hit5_count += 1
            if s_hit5: sparse_hit5_count += 1
            if h_hit5: hybrid_hit5_count += 1
            if oracle_hit5: oracle_hit5_count += 1

            if d_top1 != s_top1:
                differ_top1 += 1
                if s_hit1 and not d_hit1:
                    sparse_only_top1_gold += 1
                elif d_hit1 and not s_hit1:
                    dense_only_top1_gold += 1
                elif d_hit1 and s_hit1:
                    both_top1_gold += 1
                else:
                    neither_top1_gold += 1

        lexical_audit = {
            "n_queries": n_tune,
            "queries_with_sparse_dense_top1_differ": differ_top1,
            "differ_fraction": round(differ_top1 / n_tune, 4),
            "distribution_when_differ": {
                "dense_only_has_gold_top1": dense_only_top1_gold,
                "sparse_only_has_gold_top1": sparse_only_top1_gold,
                "both_have_gold_top1": both_top1_gold,
                "neither_has_gold_top1": neither_top1_gold,
            },
            "hit_at_1": {
                "dense": round(dense_hit1_count / n_tune, 4),
                "sparse": round(sparse_hit1_count / n_tune, 4),
                "hybrid": round(hybrid_hit1_count / n_tune, 4),
                "oracle_best_channel": round(oracle_hit1_count / n_tune, 4),
            },
            "hit_at_5": {
                "dense": round(dense_hit5_count / n_tune, 4),
                "sparse": round(sparse_hit5_count / n_tune, 4),
                "hybrid": round(hybrid_hit5_count / n_tune, 4),
                "oracle_best_channel": round(oracle_hit5_count / n_tune, 4),
            }
        }
        print(f"Top-1 Differing Queries: {differ_top1}/{n_tune} ({lexical_audit['differ_fraction']*100:.1f}%)")
        print(f"When differing: Dense has gold={dense_only_top1_gold}, Sparse has gold={sparse_only_top1_gold}, Neither={neither_top1_gold}")
        print(f"Hit@1: Dense={lexical_audit['hit_at_1']['dense']:.4f}, Sparse={lexical_audit['hit_at_1']['sparse']:.4f}, Hybrid={lexical_audit['hit_at_1']['hybrid']:.4f}, Oracle Bound={lexical_audit['hit_at_1']['oracle_best_channel']:.4f}")
        print(f"Hit@5: Dense={lexical_audit['hit_at_5']['dense']:.4f}, Sparse={lexical_audit['hit_at_5']['sparse']:.4f}, Hybrid={lexical_audit['hit_at_5']['hybrid']:.4f}, Oracle Bound={lexical_audit['hit_at_5']['oracle_best_channel']:.4f}")

    # -------------------------------------------------------------
    # 4. Context Precision Headroom (50 Scored BENCH Queries)
    # -------------------------------------------------------------
    print("\n--- 4. Context Precision Headroom Analysis (N=50 Scored BENCH Queries) ---")
    scores_rows = []
    with open(PER_QUERY_SCORES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores_rows.append(row)

    n_scored = len(scores_rows)
    cp_headroom = {}
    for mode in ["dense", "hybrid", "prismx"]:
        cp_key = f"{mode}_cp"
        less_than_1 = 0
        first_useful_below_rank1 = 0
        for r in scores_rows:
            cp_val = float(r[cp_key])
            if cp_val < 1.0 - 1e-5:
                less_than_1 += 1
                # Reconstruct verdicts
                verdicts = find_verdicts_for_cp(cp_val)
                # If rank 1 verdict is False (0) and there is at least one useful passage
                if verdicts[0] == 0 and sum(verdicts) > 0:
                    first_useful_below_rank1 += 1

        cp_headroom[mode] = {
            "n_queries": n_scored,
            "queries_cp_less_than_1": less_than_1,
            "fraction_cp_less_than_1": round(less_than_1 / n_scored, 4),
            "queries_first_useful_below_rank1": first_useful_below_rank1,
            "fraction_first_useful_below_rank1": round(first_useful_below_rank1 / n_scored, 4),
        }
        print(f"Mode '{mode}': CP < 1.0 in {less_than_1}/{n_scored} ({less_than_1/n_scored*100:.1f}%) | First useful below rank 1: {first_useful_below_rank1}/{n_scored} ({first_useful_below_rank1/n_scored*100:.1f}%)")

    # -------------------------------------------------------------
    # 5. Top 10 Largest Rerank-vs-Hybrid CP Drops
    # -------------------------------------------------------------
    print("\n--- 5. Top 10 Largest Rerank-vs-Hybrid CP Drops ---")
    drops = []
    text_store = TextStore(db_path=str(DB_PATH))

    bench_c100k_map = {str(q["query_id"]): q for q in bench_c100k}

    for r in scores_rows:
        qid = str(r["query_id"])
        h_cp = float(r["hybrid_cp"])
        r_cp = float(r["prismx_cp"])
        delta = round(h_cp - r_cp, 4)
        drops.append({
            "query_id": qid,
            "query": r["query"],
            "reference_answer": r["reference_answer"],
            "hybrid_cp": h_cp,
            "rerank_cp": r_cp,
            "delta_drop": delta,
        })

    drops.sort(key=lambda x: x["delta_drop"], reverse=True)
    top10_drops = drops[:10]

    top10_detailed = []
    for rank_idx, item in enumerate(top10_drops, 1):
        qid = item["query_id"]
        h_verdicts = find_verdicts_for_cp(item["hybrid_cp"])
        r_verdicts = find_verdicts_for_cp(item["rerank_cp"])

        q_ret = bench_c100k_map.get(qid, {})
        h_pids = q_ret.get("hybrid_pids", [])[:5]
        r_pids = q_ret.get("rerank_pids", [])[:5]

        # Hydrate texts
        all_pids = list(set(h_pids + r_pids))
        passage_texts = text_store.get_passages_by_ids(all_pids)

        detail_entry = {
            "rank": rank_idx,
            "query_id": qid,
            "query": item["query"],
            "reference_answer": item["reference_answer"],
            "hybrid_cp": item["hybrid_cp"],
            "rerank_cp": item["rerank_cp"],
            "delta_drop": item["delta_drop"],
            "hybrid_passages": [
                {
                    "rank": i + 1,
                    "passage_id": pid,
                    "verdict": h_verdicts[i] if i < len(h_verdicts) else None,
                    "text": passage_texts.get(pid, {}).get("text", "")[:120] + "..."
                } for i, pid in enumerate(h_pids)
            ],
            "rerank_passages": [
                {
                    "rank": i + 1,
                    "passage_id": pid,
                    "verdict": r_verdicts[i] if i < len(r_verdicts) else None,
                    "text": passage_texts.get(pid, {}).get("text", "")[:120] + "..."
                } for i, pid in enumerate(r_pids)
            ]
        }
        top10_detailed.append(detail_entry)
        print(f"Drop #{rank_idx} [QID {qid}] Delta = {item['delta_drop']:+.4f} (Hybrid CP={item['hybrid_cp']:.4f} -> Rerank CP={item['rerank_cp']:.4f})")
        print(f"  Query: \"{item['query']}\"")
        print(f"  Reference: \"{item['reference_answer']}\"")
        print(f"  Hybrid top-1 [PID {h_pids[0] if h_pids else 'None'}, verdict={h_verdicts[0]}]: {passage_texts.get(h_pids[0] if h_pids else '', {}).get('text', '')[:90]}...")
        print(f"  Rerank top-1 [PID {r_pids[0] if r_pids else 'None'}, verdict={r_verdicts[0]}]: {passage_texts.get(r_pids[0] if r_pids else '', {}).get('text', '')[:90]}...")

    # -------------------------------------------------------------
    # 6. Bootstrap SE & Sample Size N for +/-0.02 CI
    # -------------------------------------------------------------
    print("\n--- 6. Bootstrap SE & Power Analysis for Target CI (+/-0.02) ---")
    h_scores = [float(r["hybrid_cp"]) for r in scores_rows]
    d_scores = [float(r["dense_cp"]) for r in scores_rows]
    r_scores = [float(r["prismx_cp"]) for r in scores_rows]

    diff_hd = np.array(h_scores) - np.array(d_scores)
    diff_rh = np.array(r_scores) - np.array(h_scores)
    diff_rd = np.array(r_scores) - np.array(d_scores)

    rng = np.random.default_rng(42)
    n_resamples = 10000
    n_curr = len(diff_hd)

    boot_hd = np.mean(rng.choice(diff_hd, size=(n_resamples, n_curr), replace=True), axis=1)
    boot_rh = np.mean(rng.choice(diff_rh, size=(n_resamples, n_curr), replace=True), axis=1)
    boot_rd = np.mean(rng.choice(diff_rd, size=(n_resamples, n_curr), replace=True), axis=1)

    se_hd = float(np.std(boot_hd))
    se_rh = float(np.std(boot_rh))
    se_rd = float(np.std(boot_rd))

    std_hd = float(np.std(diff_hd, ddof=1))
    std_rh = float(np.std(diff_rh, ddof=1))
    std_rd = float(np.std(diff_rd, ddof=1))

    # Target half-width = 0.02 -> 1.96 * (std / sqrt(N)) = 0.02 -> N = (1.96 * std / 0.02)^2
    n_needed_hd = math.ceil((1.96 * std_hd / 0.02) ** 2)
    n_needed_rh = math.ceil((1.96 * std_rh / 0.02) ** 2)
    n_needed_rd = math.ceil((1.96 * std_rd / 0.02) ** 2)

    power_stats = {
        "hybrid_minus_dense": {
            "mean_delta": round(float(np.mean(diff_hd)), 4),
            "sample_std": round(std_hd, 4),
            "bootstrap_se": round(se_hd, 4),
            "current_95_ci_halfwidth": round(1.96 * se_hd, 4),
            "n_needed_for_pm_0_02_ci": n_needed_hd,
        },
        "rerank_minus_hybrid": {
            "mean_delta": round(float(np.mean(diff_rh)), 4),
            "sample_std": round(std_rh, 4),
            "bootstrap_se": round(se_rh, 4),
            "current_95_ci_halfwidth": round(1.96 * se_rh, 4),
            "n_needed_for_pm_0_02_ci": n_needed_rh,
        },
        "rerank_minus_dense": {
            "mean_delta": round(float(np.mean(diff_rd)), 4),
            "sample_std": round(std_rd, 4),
            "bootstrap_se": round(se_rd, 4),
            "current_95_ci_halfwidth": round(1.96 * se_rd, 4),
            "n_needed_for_pm_0_02_ci": n_needed_rd,
        }
    }

    print(f"Hybrid - Dense : Mean Delta = {power_stats['hybrid_minus_dense']['mean_delta']:+.4f}, SE = {se_hd:.4f} (95% CI +/-{1.96*se_hd:.4f}). N needed for +/-0.02: {n_needed_hd}")
    print(f"Rerank - Hybrid: Mean Delta = {power_stats['rerank_minus_hybrid']['mean_delta']:+.4f}, SE = {se_rh:.4f} (95% CI +/-{1.96*se_rh:.4f}). N needed for +/-0.02: {n_needed_rh}")
    print(f"Rerank - Dense : Mean Delta = {power_stats['rerank_minus_dense']['mean_delta']:+.4f}, SE = {se_rd:.4f} (95% CI +/-{1.96*se_rd:.4f}). N needed for +/-0.02: {n_needed_rd}")

    # -------------------------------------------------------------
    # 7. Write Combined Output
    # -------------------------------------------------------------
    final_output = {
        "bench_retrieval_overlap": {
            "hybrid_vs_dense": bench_hd,
            "rerank_vs_hybrid": bench_hr,
            "rerank_vs_dense": bench_dr,
        },
        "tune_retrieval_overlap": {
            "hybrid_vs_dense": tune_hd,
            "rerank_vs_hybrid": tune_hr,
            "rerank_vs_dense": tune_dr,
        },
        "lexical_contribution_audit": lexical_audit,
        "context_precision_headroom": cp_headroom,
        "top_10_rerank_vs_hybrid_drops": top10_detailed,
        "power_analysis": power_stats,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved comprehensive zero-token audit to {OUT_JSON}")


if __name__ == "__main__":
    main()
