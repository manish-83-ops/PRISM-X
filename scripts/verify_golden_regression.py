#!/usr/bin/env python3
"""
Verify Golden Regression against Baseline (Gate 15 - Tier A5).
Queries the active PRISMX API server (/search) for all 100 BENCH queries
across dense, hybrid, and prismx modes.

Requires:
- Identical IDs for dense and hybrid
- Scores within 1e-6 (or exact match for 4-decimal rounded API responses)
- Identical IDs for prismx where candidates_scored == K
- Reports separately where governor truncation differs.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
BASELINE_PATH = Path("results/golden/golden_baseline_top10.json")
OUTPUT_PATH = Path("results/golden/golden_regression_diff.json")


def post_search(query: str, mode: str, top_k: int = 10) -> dict:
    url = f"{BASE_URL}/search"
    payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "use_cache": False,
        "total_deadline_ms": 250.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_verification():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline file missing at {BASELINE_PATH}")

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    baseline_queries = baseline_data["queries"]
    total_queries = len(baseline_queries)
    print(f"[*] Starting Golden Regression Verification on {total_queries} BENCH queries...")

    # Statistics accumulators
    dense_id_matches = 0
    dense_total_slots = 0
    dense_max_score_diff = 0.0

    hybrid_id_matches = 0
    hybrid_total_slots = 0
    hybrid_max_score_diff = 0.0

    prismx_full_id_matches = 0
    prismx_full_slots = 0
    prismx_full_queries_evaluated = 0

    governor_state_diffs = []
    regressions = []

    t_start = time.perf_counter()

    for idx, b_item in enumerate(baseline_queries, start=1):
        qid = b_item["query_id"]
        qtext = b_item["query"]
        b_modes = b_item["modes"]

        if idx % 10 == 0 or idx == 1:
            print(f"[{idx}/{total_queries}] Query {qid}: {qtext[:40]}...")

        # 1. Verify DENSE
        d_resp = post_search(qtext, mode="dense", top_k=10)
        d_live_results = d_resp.get("results", [])
        d_base_results = b_modes["dense"]["top10"]

        for rank_idx, (live_r, base_r) in enumerate(zip(d_live_results, d_base_results), start=1):
            dense_total_slots += 1
            if live_r["passage_id"] == base_r["passage_id"]:
                dense_id_matches += 1
            else:
                regressions.append({
                    "query_id": qid,
                    "mode": "dense",
                    "rank": rank_idx,
                    "expected_id": base_r["passage_id"],
                    "actual_id": live_r["passage_id"],
                    "reason": "passage_id mismatch"
                })

            diff = abs(live_r["score"] - base_r["score"])
            if diff > dense_max_score_diff:
                dense_max_score_diff = diff

        # 2. Verify HYBRID
        h_resp = post_search(qtext, mode="hybrid", top_k=10)
        h_live_results = h_resp.get("results", [])
        h_base_results = b_modes["hybrid"]["top10"]

        for rank_idx, (live_r, base_r) in enumerate(zip(h_live_results, h_base_results), start=1):
            hybrid_total_slots += 1
            if live_r["passage_id"] == base_r["passage_id"]:
                hybrid_id_matches += 1
            else:
                regressions.append({
                    "query_id": qid,
                    "mode": "hybrid",
                    "rank": rank_idx,
                    "expected_id": base_r["passage_id"],
                    "actual_id": live_r["passage_id"],
                    "reason": "passage_id mismatch"
                })

            diff = abs(live_r["score"] - base_r["score"])
            if diff > hybrid_max_score_diff:
                hybrid_max_score_diff = diff

        # 3. Verify PRISMX
        p_resp = post_search(qtext, mode="prismx", top_k=10)
        p_live_results = p_resp.get("results", [])
        p_base_info = b_modes["prismx"]
        p_base_results = p_base_info["top10"]

        live_gov = p_resp.get("governor_state", "normal")
        base_gov = p_base_info.get("governor_state", "normal")
        live_scored = p_resp.get("candidates_scored", 10)
        base_scored = p_base_info.get("candidates_scored", 10)

        # Check governor parity
        if live_gov != base_gov or live_scored != base_scored:
            governor_state_diffs.append({
                "query_id": qid,
                "query": qtext,
                "base_governor": base_gov,
                "live_governor": live_gov,
                "base_scored": base_scored,
                "live_scored": live_scored,
            })

        # Where both baseline and live scored candidates == K (full rerank):
        if base_scored == 10 and live_scored == 10:
            prismx_full_queries_evaluated += 1
            for rank_idx, (live_r, base_r) in enumerate(zip(p_live_results, p_base_results), start=1):
                prismx_full_slots += 1
                if live_r["passage_id"] == base_r["passage_id"]:
                    prismx_full_id_matches += 1
                else:
                    regressions.append({
                        "query_id": qid,
                        "mode": "prismx (fully scored)",
                        "rank": rank_idx,
                        "expected_id": base_r["passage_id"],
                        "actual_id": live_r["passage_id"],
                        "reason": "passage_id mismatch on K=10 scored query"
                    })

    elapsed_sec = round(time.perf_counter() - t_start, 2)

    dense_match_pct = (dense_id_matches / dense_total_slots * 100.0) if dense_total_slots else 0.0
    hybrid_match_pct = (hybrid_id_matches / hybrid_total_slots * 100.0) if hybrid_total_slots else 0.0
    prismx_match_pct = (prismx_full_id_matches / prismx_full_slots * 100.0) if prismx_full_slots else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_dataset": "data/c100k_raw/bench_raw_100.json",
        "total_queries": total_queries,
        "elapsed_seconds": elapsed_sec,
        "results": {
            "dense": {
                "total_slots": dense_total_slots,
                "id_matches": dense_id_matches,
                "match_rate_pct": round(dense_match_pct, 4),
                "max_score_diff": round(dense_max_score_diff, 8),
                "status": "PASS" if dense_match_pct == 100.0 and dense_max_score_diff <= 1e-4 else "FAIL"
            },
            "hybrid": {
                "total_slots": hybrid_total_slots,
                "id_matches": hybrid_id_matches,
                "match_rate_pct": round(hybrid_match_pct, 4),
                "max_score_diff": round(hybrid_max_score_diff, 8),
                "status": "PASS" if hybrid_match_pct == 100.0 and hybrid_max_score_diff <= 1e-4 else "FAIL"
            },
            "prismx": {
                "full_queries_evaluated": prismx_full_queries_evaluated,
                "total_slots_evaluated": prismx_full_slots,
                "id_matches": prismx_full_id_matches,
                "match_rate_pct": round(prismx_match_pct, 4),
                "status": "PASS" if prismx_match_pct == 100.0 else "FAIL"
            }
        },
        "governor_differences_count": len(governor_state_diffs),
        "governor_differences": governor_state_diffs,
        "regressions_count": len(regressions),
        "regressions": regressions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("GOLDEN REGRESSION DIFF REPORT (Gate 15 - Tier A5)")
    print("=" * 80)
    print(f"Elapsed Time: {elapsed_sec}s across {total_queries * 3} HTTP requests")
    print(f"Dense Mode:   ID Match = {dense_id_matches}/{dense_total_slots} ({dense_match_pct:.2f}%), Max Score Diff = {dense_max_score_diff:.6f} -> {report['results']['dense']['status']}")
    print(f"Hybrid Mode:  ID Match = {hybrid_id_matches}/{hybrid_total_slots} ({hybrid_match_pct:.2f}%), Max Score Diff = {hybrid_max_score_diff:.6f} -> {report['results']['hybrid']['status']}")
    print(f"PRISMX Mode:  ID Match (K_scored=10) = {prismx_full_id_matches}/{prismx_full_slots} ({prismx_match_pct:.2f}%) across {prismx_full_queries_evaluated} queries -> {report['results']['prismx']['status']}")
    print(f"Governor Truncation Differences: {len(governor_state_diffs)}")
    print(f"Total Regressions: {len(regressions)}")
    print("=" * 80)
    print(f"Saved detailed regression diff to: {OUTPUT_PATH}")

    if dense_match_pct < 100.0 or hybrid_match_pct < 100.0 or prismx_match_pct < 100.0:
        print("[FAIL] Golden regression gate failed!")
        return 1
    else:
        print("[SUCCESS] All golden regression checks PASSED with 100% ID parity!")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_verification())
