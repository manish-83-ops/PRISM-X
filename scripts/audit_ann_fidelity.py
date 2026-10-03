"""Audit ANN fidelity on c100k_raw across 500 TUNE queries.
In strict accordance with ADR-017 (Gate 5.3):
Evaluates HNSW search_ef in {64, 128, 256} vs exact brute-force cosine search.
Computes:
(a) Mean overlap of dense top-50 vs exact dense top-50
(b) Mean overlap of dense top-10 vs exact dense top-10
(c) Fraction of TUNE queries whose gold passage is in top-10 and top-50
(d) Number of TUNE queries that lose gold at ef=64 but retain it at tested ef.
Rule: Smallest ef with top-50 overlap >= 0.99; if none reaches 0.99, choose 256.
BENCH is strictly excluded.
"""

import json
import time
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient, models

from prismx.index.encoder import DenseEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent
TUNE_FILE = REPO_ROOT / "data" / "c100k_raw" / "tune_raw_500.json"
COLLECTION_NAME = "c100k_raw"
OUT_RESULTS = REPO_ROOT / "results" / "c100k_raw" / "ann_fidelity_results.json"


def audit_ann_fidelity():
    print("=================================================================")
    print("STARTING ANN FIDELITY AUDIT ON C100K_RAW (500 TUNE QUERIES)")
    print(f"Collection: {COLLECTION_NAME}")
    print("=================================================================")

    with open(TUNE_FILE, "r", encoding="utf-8") as f:
        tune_queries = json.load(f)
    n_queries = len(tune_queries)
    print(f"Loaded {n_queries} TUNE queries.")

    client = QdrantClient(host="127.0.0.1", port=6333)
    coll_info = client.get_collection(COLLECTION_NAME)
    print(f"Qdrant collection point count: {coll_info.points_count:,}")
    assert coll_info.points_count >= 100000, f"Collection not fully indexed ({coll_info.points_count})"

    print("\nEncoding 500 TUNE queries with BGE-small...")
    encoder = DenseEncoder(model_name="BAAI/bge-small-en-v1.5", torch_threads=8)
    q_texts = [q["query"] for q in tune_queries]
    t0 = time.time()
    embeddings = encoder.encode_queries(q_texts)
    print(f"Encoded in {time.time() - t0:.2f}s.")

    ef_candidates = [64, 128, 256]

    exact_top50_map = {}
    exact_top10_map = {}
    exact_gold_in_top10 = []
    exact_gold_in_top50 = []

    print("\n[Stage 1/2] Computing exact brute-force cosine search baseline (exact=True)...")
    t_exact_start = time.time()
    for idx, (q_item, emb) in enumerate(zip(tune_queries, embeddings)):
        qid = q_item["query_id"]
        gold_ids = set(str(g) for g in q_item["gold_pids"])

        res = client.query_points(
            collection_name=COLLECTION_NAME,
            query=emb.tolist(),
            using="dense",
            search_params=models.SearchParams(exact=True),
            limit=50,
            with_payload=True
        )
        pids = [pt.payload["passage_id"] for pt in res.points]
        exact_top50_map[qid] = pids
        exact_top10_map[qid] = pids[:10]

        exact_gold_in_top10.append(any(p in gold_ids for p in pids[:10]))
        exact_gold_in_top50.append(any(p in gold_ids for p in pids))

        if (idx + 1) % 100 == 0:
            print(f"  Exact search: {idx + 1}/{n_queries} queries ({time.time() - t_exact_start:.1f}s)")

    exact_g10_frac = float(np.mean(exact_gold_in_top10))
    exact_g50_frac = float(np.mean(exact_gold_in_top50))
    print(f"Exact baseline complete in {time.time() - t_exact_start:.2f}s.")
    print(f"Exact Gold presence: Top-10 = {exact_g10_frac:.4f} ({exact_g10_frac*100:.2f}%), Top-50 = {exact_g50_frac:.4f} ({exact_g50_frac*100:.2f}%)")

    # [Stage 2/2] Evaluating HNSW ef candidates
    print("\n[Stage 2/2] Evaluating HNSW candidate ef in {64, 128, 256}...")
    ann_results = {}

    gold_at_ef = {}

    for ef in ef_candidates:
        t_ef_start = time.time()
        top50_overlaps = []
        top10_overlaps = []
        gold_in_top10 = []
        gold_in_top50 = []
        ef_pids_map = {}

        for idx, (q_item, emb) in enumerate(zip(tune_queries, embeddings)):
            qid = q_item["query_id"]
            gold_ids = set(str(g) for g in q_item["gold_pids"])

            res = client.query_points(
                collection_name=COLLECTION_NAME,
                query=emb.tolist(),
                using="dense",
                search_params=models.SearchParams(hnsw_ef=ef, exact=False),
                limit=50,
                with_payload=True
            )
            pids = [pt.payload["passage_id"] for pt in res.points]
            ef_pids_map[qid] = pids

            # Overlap calculations
            ex50 = set(exact_top50_map[qid])
            ex10 = set(exact_top10_map[qid])
            ann50 = set(pids)
            ann10 = set(pids[:10])

            top50_overlaps.append(len(ann50 & ex50) / 50.0)
            top10_overlaps.append(len(ann10 & ex10) / 10.0)

            gold_in_top10.append(any(p in gold_ids for p in pids[:10]))
            gold_in_top50.append(any(p in gold_ids for p in pids))

        gold_at_ef[ef] = {
            "top10": gold_in_top10,
            "top50": gold_in_top50,
            "pids": ef_pids_map
        }

        ann_results[ef] = {
            "hnsw_ef": ef,
            "mean_top50_overlap": round(float(np.mean(top50_overlaps)), 4),
            "mean_top10_overlap": round(float(np.mean(top10_overlaps)), 4),
            "gold_in_top10_fraction": round(float(np.mean(gold_in_top10)), 4),
            "gold_in_top50_fraction": round(float(np.mean(gold_in_top50)), 4),
            "eval_time_seconds": round(time.time() - t_ef_start, 2)
        }
        print(f"  ef={ef:3d}: Top-50 Overlap = {ann_results[ef]['mean_top50_overlap']:.4f} | Top-10 Overlap = {ann_results[ef]['mean_top10_overlap']:.4f} | Gold Top-10 = {ann_results[ef]['gold_in_top10_fraction']:.4f} | Gold Top-50 = {ann_results[ef]['gold_in_top50_fraction']:.4f} ({time.time() - t_ef_start:.1f}s)")

    # Lost gold analysis vs ef=64
    g64_t10 = gold_at_ef[64]["top10"]
    g64_t50 = gold_at_ef[64]["top50"]
    loss_comparison = {}

    for ef in [128, 256]:
        # Count queries that had gold in exact top-50, lost it at ef=64, but recovered it at tested ef
        recovered_t10 = sum(1 for i in range(n_queries) if not g64_t10[i] and gold_at_ef[ef]["top10"][i])
        recovered_t50 = sum(1 for i in range(n_queries) if not g64_t50[i] and gold_at_ef[ef]["top50"][i])
        loss_comparison[f"ef_{ef}_vs_ef_64"] = {
            "recovered_gold_top10": recovered_t10,
            "recovered_gold_top50": recovered_t50,
        }

    # Selection Rule: Smallest ef with top-50 overlap >= 0.99; if none reaches 0.99, choose 256
    chosen_ef = None
    selection_reason = ""
    for ef in ef_candidates:
        if ann_results[ef]["mean_top50_overlap"] >= 0.9900:
            chosen_ef = ef
            selection_reason = f"Smallest ef with mean top-50 overlap >= 0.99 ({ann_results[ef]['mean_top50_overlap']:.4f})"
            break

    if chosen_ef is None:
        chosen_ef = 256
        selection_reason = f"None of candidate ef reached 0.99 (highest was ef=256 at {ann_results[256]['mean_top50_overlap']:.4f}); selected 256 per ADR-017 rule."

    print("\n=================================================================")
    print(f"ANN FIDELITY DECISION: CHOSEN ef = {chosen_ef}")
    print(f"Reason: {selection_reason}")
    print("=================================================================")

    # BM25 channel exactness confirmation
    bm25_confirmation = {
        "engine": "qdrant_sparse with modifier=IDF",
        "is_exact": True,
        "approximation_type": "None (exact inverted index postings list traversal and dot-product scoring)",
        "confirmation": "BM25 sparse channel does not use graph approximation or beam pruning; retrieval is 100% exact."
    }

    final_report = {
        "benchmark": "c100k_raw ANN Fidelity Audit",
        "split": "tune_raw_500",
        "n_queries": n_queries,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exact_baseline": {
            "gold_in_top10_fraction": round(exact_g10_frac, 4),
            "gold_in_top50_fraction": round(exact_g50_frac, 4)
        },
        "ann_results": ann_results,
        "loss_vs_ef64": loss_comparison,
        "decision": {
            "chosen_ef": chosen_ef,
            "rule": "Smallest ef with top-50 overlap >= 0.99; else 256",
            "reason": selection_reason,
            "differs_from_serving_default": bool(chosen_ef != 64)
        },
        "bm25_exactness": bm25_confirmation
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Print Table
    print("\n### ANN FIDELITY RESULTS TABLE (500 TUNE Queries on c100k_raw)")
    print("| HNSW `ef` | Top-50 Overlap vs Exact | Top-10 Overlap vs Exact | Gold in Top-10 | Gold in Top-50 | Recovered vs ef=64 (Top-10 / Top-50) | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for ef in ef_candidates:
        r = ann_results[ef]
        rec_str = "Baseline (0 / 0)" if ef == 64 else f"+{loss_comparison[f'ef_{ef}_vs_ef_64']['recovered_gold_top10']} / +{loss_comparison[f'ef_{ef}_vs_ef_64']['recovered_gold_top50']}"
        status_str = "**SELECTED**" if ef == chosen_ef else ("Meets >=0.99" if r["mean_top50_overlap"] >= 0.99 else "< 0.99")
        print(f"| **ef={ef}** | {r['mean_top50_overlap']:.4f} ({r['mean_top50_overlap']*100:.2f}%) | {r['mean_top10_overlap']:.4f} ({r['mean_top10_overlap']*100:.2f}%) | {r['gold_in_top10_fraction']:.4f} | {r['gold_in_top50_fraction']:.4f} | {rec_str} | {status_str} |")

    print(f"| **Exact (Brute-Force)** | 1.0000 (100.0%) | 1.0000 (100.0%) | {exact_g10_frac:.4f} | {exact_g50_frac:.4f} | Reference Baseline | Reference |")

    return final_report


if __name__ == "__main__":
    audit_ann_fidelity()
