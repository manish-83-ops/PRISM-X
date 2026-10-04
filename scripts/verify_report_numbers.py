#!/usr/bin/env python3
"""
PRISMX Claims-to-Files Verification and Audit Tool
Recomputes, from results/ files only, every headline number in the report and README.
Outputs audit table: claim | report_value | file | recomputed_value | status
"""

import json
import csv
from pathlib import Path
import pandas as pd

def run_audit():
    results = []

    def check(claim, report_val, file_path, recomputed_val, tolerance=0.001):
        if file_path is None or not Path(file_path).exists():
            status = "NO SOURCE FILE"
            recomp_str = "N/A (File Missing)"
        else:
            recomp_str = str(recomputed_val)
            if isinstance(report_val, (int, float)) and isinstance(recomputed_val, (int, float)):
                match = abs(float(report_val) - float(recomputed_val)) <= tolerance
            else:
                match = (str(report_val).strip() == str(recomputed_val).strip())
            status = "MATCH" if match else "MISMATCH"
        
        results.append({
            "claim": claim,
            "report_val": str(report_val),
            "file": file_path or "NONE",
            "recomputed_val": recomp_str,
            "status": status
        })

    # 1. Corpus Scale
    bpath = "results/c100k_raw/build_stats.json"
    if Path(bpath).exists():
        with open(bpath) as f:
            bdata = json.load(f)
        check("Corpus Qdrant Points", 100008, bpath, bdata.get("points_count"))
        check("Corpus SQLite Passages", 100008, bpath, bdata.get("sqlite_count"))
    else:
        check("Corpus Qdrant Points", 100008, bpath, None)
        check("Corpus SQLite Passages", 100008, bpath, None)

    # 2. BENCH Quality Metrics (N=100)
    bench_path = "results/c100k_raw/bench_eval_results.json"
    if Path(bench_path).exists():
        with open(bench_path) as f:
            bench = json.load(f)
        sm = bench.get("summary_metrics", {})
        d = sm.get("dense", {})
        h = sm.get("hybrid", {})
        r = sm.get("hybrid_rerank_k10", {})
        
        # Dense
        check("Dense MRR@10 Mean", 0.6032, bench_path, d.get("mrr10", {}).get("mean"))
        check("Dense MRR@10 CI Lower", 0.5344, bench_path, d.get("mrr10", {}).get("ci_lower"))
        check("Dense MRR@10 CI Upper", 0.6714, bench_path, d.get("mrr10", {}).get("ci_upper"))
        check("Dense NDCG@5 Mean", 0.6694, bench_path, d.get("ndcg5", {}).get("mean"))
        check("Dense Hit@1 Mean", 0.4200, bench_path, d.get("hit1", {}).get("mean"))
        check("Dense Recall@10 Mean", 0.9800, bench_path, d.get("recall10", {}).get("mean"))
        check("Dense Recall@50 Mean", 0.9900, bench_path, d.get("recall50", {}).get("mean"))

        # Hybrid
        check("Hybrid MRR@10 Mean", 0.5949, bench_path, h.get("mrr10", {}).get("mean"))
        check("Hybrid MRR@10 CI Lower", 0.5240, bench_path, h.get("mrr10", {}).get("ci_lower"))
        check("Hybrid MRR@10 CI Upper", 0.6638, bench_path, h.get("mrr10", {}).get("ci_upper"))
        check("Hybrid NDCG@5 Mean", 0.6582, bench_path, h.get("ndcg5", {}).get("mean"))
        check("Hybrid Hit@1 Mean", 0.4100, bench_path, h.get("hit1", {}).get("mean"))
        check("Hybrid Recall@10 Mean", 0.9700, bench_path, h.get("recall10", {}).get("mean"))
        check("Hybrid Recall@50 Mean", 0.9800, bench_path, h.get("recall50", {}).get("mean"))

        # Rerank K=10
        check("Rerank MRR@10 Mean", 0.6532, bench_path, r.get("mrr10", {}).get("mean"))
        check("Rerank MRR@10 CI Lower", 0.5853, bench_path, r.get("mrr10", {}).get("ci_lower"))
        check("Rerank MRR@10 CI Upper", 0.7195, bench_path, r.get("mrr10", {}).get("ci_upper"))
        check("Rerank NDCG@5 Mean", 0.7236, bench_path, r.get("ndcg5", {}).get("mean"))
        check("Rerank Hit@1 Mean", 0.4600, bench_path, r.get("hit1", {}).get("mean"))
        check("Rerank Recall@10 Mean", 0.9700, bench_path, r.get("recall10", {}).get("mean"))
        check("Rerank Recall@50 Mean", 0.9800, bench_path, r.get("recall50", {}).get("mean"))

        # Paired Gains & Wins/Losses/Ties
        pc = bench.get("paired_comparisons", {})
        rh = pc.get("rerank_minus_hybrid", {})
        check("Rerank - Hybrid MRR@10 Delta", 0.0583, bench_path, rh.get("mrr10", {}).get("mean_delta"))
        check("Rerank - Hybrid MRR@10 Wins", 35, bench_path, rh.get("mrr10", {}).get("wins"))
        check("Rerank - Hybrid MRR@10 Losses", 16, bench_path, rh.get("mrr10", {}).get("losses"))
        check("Rerank - Hybrid MRR@10 Ties", 49, bench_path, rh.get("mrr10", {}).get("ties"))

        # Governor Telemetry
        gov = bench.get("governor_telemetry", {})
        check("Governor Truncation Events", 5, bench_path, gov.get("truncation_events"))
        check("Governor Exhausted Before Batch1", 0, bench_path, gov.get("exhausted_before_first_batch_events"))

    # 3. HTTP Latency Benchmarks (Idle, N=100)
    lat_path = "results/c100k_raw/latency_benchmark.json"
    if Path(lat_path).exists():
        with open(lat_path) as f:
            lat = json.load(f)
        uncached = lat.get("modes_uncached", {})
        d_lat = uncached.get("dense", {})
        h_lat = uncached.get("hybrid", {})
        p_lat = uncached.get("prismx", {})
        cache_uniq = lat.get("cache_scenarios", {}).get("all_unique", {})

        check("Dense HTTP p50", 58.40, lat_path, d_lat.get("p50_ms"))
        check("Dense HTTP p95", 107.21, lat_path, d_lat.get("p95_ms"))
        check("Hybrid HTTP p50", 67.43, lat_path, h_lat.get("p50_ms"))
        check("Hybrid HTTP p95", 89.02, lat_path, h_lat.get("p95_ms"))
        check("PRISM-X HTTP p50", 203.22, lat_path, p_lat.get("p50_ms"))
        check("PRISM-X HTTP p95", 306.39, lat_path, p_lat.get("p95_ms"))
        check("PRISM-X Max Latency", 1239.35, lat_path, p_lat.get("max_ms"))
        check("Cache All-Unique p95", 250.50, lat_path, cache_uniq.get("p95_ms"))

    # 4. Raw Latency CSV (PRISM-X Stage breakdown)
    csv_path = "results/c100k_raw/raw_latency_prismx_bench100.csv"
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        rerank_p50 = round(float(df["server_rerank_ms"].quantile(0.50)), 2)
        rerank_p95 = round(float(df["server_rerank_ms"].quantile(0.95)), 2)
        server_gt_250 = int((df["server_total_ms"] > 250.0).sum())
        client_gt_250 = int((df["client_wall_clock_ms"] > 250.0).sum())
        max_row = df.loc[df["client_wall_clock_ms"].idxmax()]
        max_fetch = round(float(max_row["server_fetch_text_ms"]), 2)

        check("PRISM-X Rerank Stage p50", 125.51, csv_path, rerank_p50, tolerance=0.05)
        check("PRISM-X Rerank Stage p95", 193.00, csv_path, rerank_p95, tolerance=0.05)
        check("PRISM-X server_total > 250ms Count", 16, csv_path, server_gt_250)
        check("PRISM-X client_wall_clock > 250ms Count", 18, csv_path, client_gt_250)
        check("PRISM-X Max Outlier SQLite Hydration ms", 1116.22, csv_path, max_fetch)

    # 5. ANN Fidelity Audit (ADR-017)
    ann_path = "results/c100k_raw/ann_fidelity_results.json"
    if Path(ann_path).exists():
        with open(ann_path) as f:
            ann = json.load(f)
        ar = ann.get("ann_results", {})
        check("ANN Fidelity ef=64 Overlap", 0.9869, ann_path, ar.get("64", {}).get("mean_top50_overlap"))
        check("ANN Fidelity ef=128 Overlap", 0.9954, ann_path, ar.get("128", {}).get("mean_top50_overlap"))
        check("ANN Fidelity Chosen ef", 128, ann_path, ann.get("decision", {}).get("chosen_ef"))

    # 6. Curated Partition Fusion Sweep (Phase 2)
    fuse_path = "results/phase2/fusion_tuning_tune.json"
    if Path(fuse_path).exists():
        with open(fuse_path) as f:
            fuse = json.load(f)
        flist = fuse.get("results", [])
        w8 = next((m for m in flist if m.get("alpha") == 0.8 and m.get("method") == "weighted" and m.get("norm") == "minmax"), {})
        check("Curated Fusion alpha=0.8 NDCG@5", 0.8962, fuse_path, w8.get("ndcg5"))
        check("Curated Fusion alpha=0.8 MRR@10", 0.8856, fuse_path, w8.get("mrr10"))
        check("Curated Fusion alpha=0.8 Hit@1", 0.8467, fuse_path, w8.get("hit1"))

    # 7. FR-4 Filter Overlap Claim Audit
    f_path = "results/c100k_raw/filter_verification.json"
    if Path(f_path).exists():
        with open(f_path) as f:
            fdata = json.load(f)
        check("FR-4 Pre-retrieval Filter Pass Rate", 100.0, f_path, fdata.get("pass_rate"))
        check("FR-4 Filter Out-of-Category Leaks", 0, f_path, fdata.get("leaked_passages"))
    else:
        check("FR-4 Pre-retrieval Filter Pass Rate", 100.0, f_path, None)

    # 8. RAGAS Exploratory Table (Gate 4B Curated Partition, N=25)
    ragas_path = "results/ragas/frozen25_ragas_summary.json"
    if Path(ragas_path).exists():
        with open(ragas_path) as f:
            rag = json.load(f)
        p1 = rag.get("phases", {}).get("phase1_dense", {})
        p2 = rag.get("phases", {}).get("phase2_hybrid", {})
        p3 = rag.get("phases", {}).get("phase3_rerank", {})
        check("RAGAS Curated Phase 1 CP", 0.8539, ragas_path, p1.get("context_precision", {}).get("mean"))
        check("RAGAS Curated Phase 2 CP", 0.9184, ragas_path, p2.get("context_precision", {}).get("mean"))
        check("RAGAS Curated Phase 3 CP", 0.9126, ragas_path, p3.get("context_precision", {}).get("mean"))

    # 9. TUNE vs BENCH Dataset and Label Mixup Audit (Gate 12 Part 5)
    tune_file = Path("data/c100k_raw/tune_raw_500.json")
    bench_file = Path("data/c100k_raw/bench_raw_100.json")
    tune_qids = set()
    bench_qids = set()
    if tune_file.exists():
        with open(tune_file, "r", encoding="utf-8") as f:
            t_raw = json.load(f)
            tune_qids = {str(item.get("query_id")) for item in t_raw}
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            b_raw = json.load(f)
            bench_qids = {str(item.get("query_id")) for item in b_raw}

    check("TUNE Query Count (exactly 500)", 500, str(tune_file), len(tune_qids))
    check("BENCH Query Count (exactly 100)", 100, str(bench_file), len(bench_qids))
    overlap = len(tune_qids.intersection(bench_qids))
    check("TUNE and BENCH Disjointness (0 overlap)", 0, str(bench_file), overlap)

    # Check for label mixups in results directory
    tune_retrievals = Path("results/tune_per_query_retrievals.json")
    if tune_retrievals.exists():
        with open(tune_retrievals, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        t_count = len(t_data) if isinstance(t_data, list) else len(t_data.get("queries", []))
        check("TUNE Retrievals Query Count (must be 500)", 500, str(tune_retrievals), t_count)

    bench_eval = Path("results/c100k_raw/bench_eval_results.json")
    if bench_eval.exists():
        with open(bench_eval, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        eval_qids = set()
        per_q = b_data.get("per_query_results", {})
        for mode_res in per_q.values():
            if isinstance(mode_res, list):
                for item in mode_res:
                    eval_qids.add(str(item.get("query_id")))
        if eval_qids:
            check("BENCH Eval Results Count (must be 100)", 100, str(bench_eval), len(eval_qids))
            # Must match bench_qids and not tune_qids
            is_pure_bench = eval_qids.issubset(bench_qids) if bench_qids else True
            check("BENCH File Contains Pure BENCH QIDs", True, str(bench_eval), is_pure_bench)

    # Print Table
    print(f"\n{'CLAIM':<42} | {'REPORT VAL':<15} | {'RECOMPUTED VAL':<18} | {'STATUS':<15} | {'SOURCE FILE'}")
    print("-" * 125)
    for r in results:
        fname = Path(r['file']).name if r['file'] != "NONE" else "NO SOURCE"
        print(f"{r['claim']:<42} | {r['report_val']:<15} | {r['recomputed_val']:<18} | {r['status']:<15} | {fname}")

    print("-" * 125)
    matches = sum(1 for r in results if r["status"] == "MATCH")
    mismatches = sum(1 for r in results if r["status"] == "MISMATCH")
    no_file = sum(1 for r in results if r["status"] == "NO SOURCE FILE")
    print(f"Summary: {matches} MATCH, {mismatches} MISMATCH, {no_file} NO SOURCE FILE (Total: {len(results)})")

if __name__ == "__main__":
    run_audit()
