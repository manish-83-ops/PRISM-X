"""Gate 4A: Score BENCH once for frozen config (dense, hybrid, hybrid+rerank) with bootstrap significance testing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
import numpy as np

from prismx.config import load_config
from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore
from prismx.index.text_store import TextStore
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.rerank import CrossEncoderReranker
from prismx.retrieve.service import SearchService
from prismx.schemas import SearchRequest
from prismx.eval.metrics import (
    hit_at_1,
    success_at_k,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    dup_aware_hit_at_1,
    dup_aware_recall_at_5,
)
from prismx.eval.bootstrap import bootstrap_ci, paired_bootstrap_difference
from prismx.eval.ragas_eval import compute_non_llm_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_bench_gate4a")

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_query_list(
    service: SearchService,
    queries: list[dict],
    mode: str,
    top_k: int = 10,
    rerank: bool = False,
    rerank_k: int = 30,
) -> tuple[list[dict], dict[str, list[float]], list[float]]:
    query_records = []
    latencies = []
    per_query_metrics: dict[str, list[float]] = {
        "hit_at_1": [],
        "mrr_at_10": [],
        "ndcg_at_5": [],
        "ndcg_at_10": [],
        "recall_at_5": [],
        "recall_at_10": [],
        "success_at_5": [],
        "success_at_10": [],
        "rank_based_cp_at_5": [],
        "rank_based_cr_at_5": [],
        "rank_based_cr_at_10": [],
    }

    near_dups_file = REPO_ROOT / "data" / "manifests" / "near_duplicates_manifest.json"
    near_dups_map = {}
    if near_dups_file.exists():
        with open(near_dups_file, "r", encoding="utf-8") as f:
            near_dups_map = json.load(f).get("near_duplicates_by_gold_id", {})

    for idx, item in enumerate(queries, start=1):
        qid = str(item["query_id"])
        query = item["query"]
        gold_ids = set(str(g) for g in item["gold_passage_ids"])

        req = SearchRequest(
            query=query,
            mode=mode,
            top_k=top_k,
            rerank=rerank,
            rerank_k=rerank_k,
            use_cache=False,
        )

        t0 = time.perf_counter()
        resp = service.search(req)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_pids = [r.passage_id for r in resp.results]
        top5_pids = retrieved_pids[:5]
        top10_pids = retrieved_pids[:10]

        # Compute standard IR metrics
        h1 = hit_at_1(top5_pids, gold_ids)
        mrr = mrr_at_k(top10_pids, gold_ids, 10)
        n5 = ndcg_at_k(top5_pids, gold_ids, 5)
        n10 = ndcg_at_k(top10_pids, gold_ids, 10)
        r5 = recall_at_k(top5_pids, gold_ids, 5)
        r10 = recall_at_k(top10_pids, gold_ids, 10)
        s5 = success_at_k(top5_pids, gold_ids, 5)
        s10 = success_at_k(top10_pids, gold_ids, 10)

        # Compute rank-based context precision & recall (qrel-derived)
        non_llm = compute_non_llm_metrics(retrieved_pids, gold_ids, near_dups_map)
        cp5 = non_llm["context_precision"]
        cr5 = non_llm["context_recall"]
        cr10 = non_llm.get("context_recall_10", r10)

        per_query_metrics["hit_at_1"].append(h1)
        per_query_metrics["mrr_at_10"].append(mrr)
        per_query_metrics["ndcg_at_5"].append(n5)
        per_query_metrics["ndcg_at_10"].append(n10)
        per_query_metrics["recall_at_5"].append(r5)
        per_query_metrics["recall_at_10"].append(r10)
        per_query_metrics["success_at_5"].append(s5)
        per_query_metrics["success_at_10"].append(s10)
        per_query_metrics["rank_based_cp_at_5"].append(cp5)
        per_query_metrics["rank_based_cr_at_5"].append(cr5)
        per_query_metrics["rank_based_cr_at_10"].append(cr10)
        latencies.append(dt_ms)

        query_records.append({
            "idx": idx,
            "query_id": qid,
            "query": query,
            "gold_passage_ids": list(gold_ids),
            "retrieved_passage_ids": retrieved_pids,
            "client_latency_ms": round(dt_ms, 2),
            "server_latency_ms": resp.latency_ms.dict(),
            "results": [r.dict() for r in resp.results],
        })

    return query_records, per_query_metrics, latencies


def summarize_metrics(per_query: dict[str, list[float]]) -> dict[str, dict]:
    summary = {}
    for m_name, vals in per_query.items():
        ci_res = bootstrap_ci(vals, n_resamples=10000, seed=42)
        summary[m_name] = {
            "mean": round(float(np.mean(vals)), 4),
            "ci_lower": ci_res["ci_lower"],
            "ci_upper": ci_res["ci_upper"],
        }
    return summary


def main():
    print("====================================================================")
    print("GATE 4A: RIGOROUS EVALUATION OF FROZEN CONFIGS ON 100 BENCH QUERIES")
    print("Configs: (1) Dense Baseline, (2) Hybrid Optimized, (3) Hybrid + MiniLM-L6 INT8 Rerank (K=30)")
    print("====================================================================\n")

    cfg = load_config()

    # 1. Initialize stores and models
    text_store = TextStore(db_path=cfg["sqlite"]["db_path"])
    qdrant_store = QdrantStore(
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        grpc_port=cfg["qdrant"]["grpc_port"],
        prefer_grpc=cfg["qdrant"].get("prefer_grpc", True),
        collection_name=cfg["qdrant"]["collection_name"],
    )
    encoder = DenseEncoder(
        model_name=cfg["encoder"]["model_name"],
        embedding_dim=cfg["encoder"]["embedding_dim"],
        max_seq_length=cfg["encoder"]["max_seq_length"],
        torch_threads=cfg["encoder"].get("torch_threads", 12),
    )
    tokenizer = BM25Tokenizer(
        k1=cfg["lexical"]["k1"],
        b=cfg["lexical"]["b"],
        use_stemming=cfg["lexical"].get("stemming", False),
    )
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        torch_threads=cfg["encoder"].get("torch_threads", 12),
    )
    dense_retriever = DenseRetriever(encoder=encoder, qdrant_store=qdrant_store, config=cfg)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        tokenizer=tokenizer,
        qdrant_store=qdrant_store,
        config=cfg,
    )
    service = SearchService(
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        text_store=text_store,
        qdrant_store=qdrant_store,
        config=cfg,
        reranker=reranker,
        cache=None,  # Benchmarking uncached retrieval quality
    )

    # 2. Load BENCH split
    bench_path = REPO_ROOT / "data" / "manifests" / "split_bench.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    assert len(bench_queries) == 100, f"Expected 100 BENCH queries, got {len(bench_queries)}"

    # 3. Warm-up
    print("Step 0: Running 5 warm-up queries...")
    for q in bench_queries[:5]:
        service.search(SearchRequest(query=q["query"], mode="hybrid_rerank", top_k=5, rerank_k=30, use_cache=False))
    print("Warm-up complete.\n")

    # 4. Evaluate Frozen Config 1: Dense
    print("Step 1: Evaluating Frozen Config 1: Dense (100 BENCH queries)...")
    dense_records, dense_metrics, dense_lats = evaluate_query_list(
        service, bench_queries, mode="dense", top_k=10, rerank=False
    )
    dense_summary = summarize_metrics(dense_metrics)
    print(f"Dense Summary: Hit@1={dense_summary['hit_at_1']['mean']} | MRR@10={dense_summary['mrr_at_10']['mean']} | NDCG@5={dense_summary['ndcg_at_5']['mean']} | Recall@10={dense_summary['recall_at_10']['mean']}")

    # 5. Evaluate Frozen Config 2: Hybrid
    print("\nStep 2: Evaluating Frozen Config 2: Hybrid (alpha=0.8, minmax) (100 BENCH queries)...")
    hybrid_records, hybrid_metrics, hybrid_lats = evaluate_query_list(
        service, bench_queries, mode="hybrid", top_k=10, rerank=False
    )
    hybrid_summary = summarize_metrics(hybrid_metrics)
    print(f"Hybrid Summary: Hit@1={hybrid_summary['hit_at_1']['mean']} | MRR@10={hybrid_summary['mrr_at_10']['mean']} | NDCG@5={hybrid_summary['ndcg_at_5']['mean']} | Recall@10={hybrid_summary['recall_at_10']['mean']}")

    # 6. Evaluate Frozen Config 3: Hybrid + MiniLM-L6 INT8 Rerank (K=30)
    print("\nStep 3: Evaluating Frozen Config 3: Hybrid + Rerank (K=30) (100 BENCH queries)...")
    rerank_records, rerank_metrics, rerank_lats = evaluate_query_list(
        service, bench_queries, mode="hybrid_rerank", top_k=10, rerank=True, rerank_k=30
    )
    rerank_summary = summarize_metrics(rerank_metrics)
    print(f"Hybrid+Rerank Summary: Hit@1={rerank_summary['hit_at_1']['mean']} | MRR@10={rerank_summary['mrr_at_10']['mean']} | NDCG@5={rerank_summary['ndcg_at_5']['mean']} | Recall@10={rerank_summary['recall_at_10']['mean']}")

    # 7. Compute Paired Bootstrap Significance for each step
    print("\n--------------------------------------------------------------------")
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTS (10,000 resamples, 95% CIs):")
    print("--------------------------------------------------------------------")

    tracked_metrics = [
        "hit_at_1",
        "mrr_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
        "recall_at_5",
        "recall_at_10",
        "rank_based_cp_at_5",
    ]

    # Step 1: Hybrid vs Dense
    step1_diffs = {}
    print("\n[Step 1: Hybrid vs Dense]")
    for m in tracked_metrics:
        res = paired_bootstrap_difference(dense_metrics[m], hybrid_metrics[m], n_resamples=10000, seed=42)
        step1_diffs[m] = res
        print(f"  {m:<20}: delta={res['mean_diff']:+.4f} | 95% CI=[{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}] | W/L/T: {res['wins']}/{res['losses']}/{res['ties']} | Significant: {res['is_statistically_distinguishable']}")

    # Step 2: Hybrid+Rerank vs Hybrid
    step2_diffs = {}
    print("\n[Step 2: Hybrid+Rerank vs Hybrid]")
    for m in tracked_metrics:
        res = paired_bootstrap_difference(hybrid_metrics[m], rerank_metrics[m], n_resamples=10000, seed=42)
        step2_diffs[m] = res
        print(f"  {m:<20}: delta={res['mean_diff']:+.4f} | 95% CI=[{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}] | W/L/T: {res['wins']}/{res['losses']}/{res['ties']} | Significant: {res['is_statistically_distinguishable']}")

    # Step 3: Hybrid+Rerank vs Dense
    step3_diffs = {}
    print("\n[Step 3: Hybrid+Rerank vs Dense]")
    for m in tracked_metrics:
        res = paired_bootstrap_difference(dense_metrics[m], rerank_metrics[m], n_resamples=10000, seed=42)
        step3_diffs[m] = res
        print(f"  {m:<20}: delta={res['mean_diff']:+.4f} | 95% CI=[{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}] | W/L/T: {res['wins']}/{res['losses']}/{res['ties']} | Significant: {res['is_statistically_distinguishable']}")

    # 8. Save artifacts
    out_dir = REPO_ROOT / "results" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "bench_dense_retrievals.json", "w", encoding="utf-8") as f:
        json.dump(dense_records, f, indent=2)

    with open(out_dir / "bench_hybrid_retrievals.json", "w", encoding="utf-8") as f:
        json.dump(hybrid_records, f, indent=2)

    with open(out_dir / "bench_hybrid_rerank_retrievals.json", "w", encoding="utf-8") as f:
        json.dump(rerank_records, f, indent=2)

    final_metrics_payload = {
        "evaluation_name": "Gate 4A Frozen Configurations Comparison (BENCH 100 queries)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs": {
            "dense": {
                "name": "Phase 1: Naive Dense (bge-small-en-v1.5)",
                "summary": dense_summary,
            },
            "hybrid": {
                "name": "Phase 2: Hybrid (Dense + BM25, alpha=0.8, minmax)",
                "summary": hybrid_summary,
            },
            "hybrid_rerank": {
                "name": "Phase 3: Hybrid + MiniLM-L6 INT8 Rerank (K=30)",
                "summary": rerank_summary,
            },
        },
        "significance_tests": {
            "hybrid_vs_dense": step1_diffs,
            "hybrid_rerank_vs_hybrid": step2_diffs,
            "hybrid_rerank_vs_dense": step3_diffs,
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics_payload, f, indent=2)

    print(f"\nSaved all BENCH retrieval records and metrics to {out_dir}")

    text_store.close()
    qdrant_store.close()


if __name__ == "__main__":
    main()
