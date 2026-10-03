"""Gate 4B: Sweep candidate depth K in {5, 8, 10, 15, 20} under ADR-013 and evaluate Deadline Governor on TUNE."""

from __future__ import annotations

import json
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
from prismx.eval.metrics import mrr_at_k, recall_at_k, ndcg_at_k, hit_at_1

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_k_on_tune(
    service: SearchService,
    queries: list[dict],
    k_candidate: int,
    deadline_ms: float | None = 200.0,
) -> dict:
    mrr_list = []
    r5_list = []
    ndcg5_list = []
    h1_list = []
    latencies = []
    truncated_count = 0

    for item in queries:
        qid = str(item["query_id"])
        query = item["query"]
        gold_ids = set(str(g) for g in item["gold_passage_ids"])

        req = SearchRequest(
            query=query,
            mode="hybrid_rerank",
            top_k=5,
            rerank_k=k_candidate,
            deadline_ms=deadline_ms,
            use_cache=False,
        )

        t0 = time.perf_counter()
        resp = service.search(req)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.governor_state == "truncated":
            truncated_count += 1

        pids = [r.passage_id for r in resp.results]
        mrr_list.append(mrr_at_k(pids, gold_ids, 10))
        r5_list.append(recall_at_k(pids, gold_ids, 5))
        ndcg5_list.append(ndcg_at_k(pids, gold_ids, 5))
        h1_list.append(hit_at_1(pids, gold_ids))
        latencies.append(dt_ms)

    arr = np.array(latencies)
    return {
        "k_candidate": k_candidate,
        "deadline_ms": deadline_ms,
        "n_queries": len(queries),
        "ndcg5": round(float(np.mean(ndcg5_list)), 4),
        "mrr10": round(float(np.mean(mrr_list)), 4),
        "recall5": round(float(np.mean(r5_list)), 4),
        "hit1": round(float(np.mean(h1_list)), 4),
        "p50_ms": round(float(np.percentile(arr, 50, method="linear")), 2),
        "p90_ms": round(float(np.percentile(arr, 90, method="linear")), 2),
        "p95_ms": round(float(np.percentile(arr, 95, method="linear")), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "truncation_count": truncated_count,
        "truncation_rate_pct": round((truncated_count / len(queries)) * 100.0, 2),
    }


def main():
    print("====================================================================")
    print("GATE 4B: CANDIDATE DEPTH K SWEEP UNDER ADR-013 & DEADLINE GOVERNOR")
    print("====================================================================\n")

    cfg = load_config()

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
        torch_threads=cfg["encoder"].get("torch_threads", 8),
    )
    tokenizer = BM25Tokenizer(
        k1=cfg["lexical"]["k1"],
        b=cfg["lexical"]["b"],
        use_stemming=cfg["lexical"].get("stemming", False),
    )
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        torch_threads=8,
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
        cache=None,
    )

    tune_path = REPO_ROOT / "data" / "manifests" / "split_tune.json"
    with open(tune_path, "r", encoding="utf-8") as f:
        tune_queries = json.load(f)

    # Use first 150 seeded TUNE queries for evaluation
    eval_queries = tune_queries[:150]
    print(f"Loaded {len(eval_queries)} queries from TUNE split.")

    # Sweep K in {5, 8, 10, 15, 20} with 200ms deadline
    k_values = [5, 8, 10, 15, 20]
    pareto_results = {}

    print("\n--------------------------------------------------------------------")
    print(f"{'K':<5} {'NDCG@5':<8} {'MRR@10':<8} {'Hit@1':<8} {'p50(ms)':<9} {'p95(ms)':<9} {'Trunc%':<8}")
    print("--------------------------------------------------------------------")

    for k in k_values:
        res = evaluate_k_on_tune(service, eval_queries, k_candidate=k, deadline_ms=200.0)
        pareto_results[k] = res
        print(f"{k:<5} {res['ndcg5']:<8.4f} {res['mrr10']:<8.4f} {res['hit1']:<8.4f} {res['p50_ms']:<9.1f} {res['p95_ms']:<9.1f} {res['truncation_rate_pct']:<8.1f}%")

    # Sweep Deadline Governor threshold on best K
    print("\nDeadline Governor Threshold Sweep on K=10:")
    governor_sweep = {}
    for d_ms in [150.0, 200.0, 250.0, None]:
        label = f"{d_ms}ms" if d_ms else "unlimited"
        res_gov = evaluate_k_on_tune(service, eval_queries, k_candidate=10, deadline_ms=d_ms)
        governor_sweep[label] = res_gov
        print(f"  Deadline {label:<10}: NDCG@5={res_gov['ndcg5']:.4f} | p95={res_gov['p95_ms']:.1f}ms | Truncation={res_gov['truncation_rate_pct']:.1f}%")

    out_file = REPO_ROOT / "results" / "phase3" / "tune_pareto_k.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "k_sweep": pareto_results,
                "governor_sweep": governor_sweep,
            },
            f,
            indent=2,
        )
    print(f"\nSaved Pareto sweep and governor results to {out_file}")

    text_store.close()
    qdrant_store.close()


if __name__ == "__main__":
    main()
