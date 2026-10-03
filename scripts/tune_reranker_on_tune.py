"""Tune Cross-Encoder Candidate Depth K on TUNE Split (150 queries) strictly before evaluating BENCH (ADR-011)."""

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
from prismx.retrieve.cache import QueryCache
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.rerank import CrossEncoderReranker
from prismx.retrieve.service import SearchService
from prismx.schemas import SearchRequest
from prismx.eval.metrics import mrr_at_k, recall_at_k, ndcg_at_k, hit_at_1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tune_reranker")

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_rerank_k(
    service: SearchService,
    queries: list[dict],
    k_candidate: int,
) -> dict:
    mrr_list = []
    r5_list = []
    ndcg5_list = []
    h1_list = []
    latencies = []

    for item in queries:
        qid = str(item["query_id"])
        query = item["query"]
        gold_ids = set(str(g) for g in item["gold_passage_ids"])

        req = SearchRequest(
            query=query,
            mode="hybrid_rerank",
            top_k=5,
            rerank_k=k_candidate,
            use_cache=False,  # Measure uncached quality and latency
        )

        t0 = time.perf_counter()
        resp = service.search(req)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        pids = [r.passage_id for r in resp.results]
        mrr_list.append(mrr_at_k(pids, gold_ids, 10))
        r5_list.append(recall_at_k(pids, gold_ids, 5))
        ndcg5_list.append(ndcg_at_k(pids, gold_ids, 5))
        h1_list.append(hit_at_1(pids, gold_ids))
        latencies.append(dt_ms)

    return {
        "k_candidate": k_candidate,
        "n_queries": len(queries),
        "ndcg5": float(np.mean(ndcg5_list)),
        "mrr10": float(np.mean(mrr_list)),
        "recall5": float(np.mean(r5_list)),
        "hit1": float(np.mean(h1_list)),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def main():
    print("====================================================================")
    print("GATE 4A: CROSS-ENCODER RERANKER CANDIDATE DEPTH K TUNING ON TUNE SPLIT")
    print("Selection Rule: Maximize NDCG@5; tie-break (|delta| <= 0.001) to smaller K.")
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
        cache=None,
    )

    # 2. Load TUNE split
    tune_path = REPO_ROOT / "data" / "manifests" / "split_tune.json"
    with open(tune_path, "r", encoding="utf-8") as f:
        tune_queries = json.load(f)

    eval_set = tune_queries[:150]
    print(f"Loaded {len(eval_set)} queries from TUNE split.")

    # 3. Evaluate candidate depths K in {10, 20, 30}
    candidates_k = [10, 20, 30]
    results = {}

    for k in candidates_k:
        print(f"\n--> Evaluating K = {k} on {len(eval_set)} TUNE queries...")
        t0 = time.time()
        res = evaluate_rerank_k(service, eval_set, k)
        elapsed = time.time() - t0
        results[k] = res
        print(f"    K={k}: NDCG@5={res['ndcg5']:.4f} | MRR@10={res['mrr10']:.4f} | Hit@1={res['hit1']:.4f} | Mean Latency={res['mean_latency_ms']:.1f}ms | p95={res['p95_latency_ms']:.1f}ms (eval elapsed: {elapsed:.1f}s)")

    # 4. Apply Selection Rule
    print("\n--------------------------------------------------------------------")
    print("CANDIDATE DEPTH K ABLATION SUMMARY ON TUNE:")
    print("--------------------------------------------------------------------")
    print(f"{'K':<6} {'NDCG@5':<10} {'MRR@10':<10} {'Hit@1':<10} {'p95 Latency':<14}")
    for k in candidates_k:
        r = results[k]
        print(f"{k:<6} {r['ndcg5']:<10.4f} {r['mrr10']:<10.4f} {r['hit1']:<10.4f} {r['p95_latency_ms']:<14.1f}")

    # Determine winner
    best_k = candidates_k[0]
    best_ndcg = results[best_k]["ndcg5"]

    for k in candidates_k[1:]:
        ndcg_k = results[k]["ndcg5"]
        if ndcg_k > best_ndcg + 0.0010:
            best_k = k
            best_ndcg = ndcg_k
        elif abs(ndcg_k - best_ndcg) <= 0.0010:
            # Tie within 0.0010 -> pick smaller K
            if k < best_k:
                best_k = k
                best_ndcg = ndcg_k

    print(f"\n>>> SELECTION RULE WINNER: K = {best_k} (NDCG@5 = {results[best_k]['ndcg5']:.4f})")

    out_dir = REPO_ROOT / "results" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "tune_reranker_k.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_rule": "Maximize NDCG@5 on TUNE; tie-break (|delta| <= 0.001) to smaller K",
                "chosen_k": best_k,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved TUNE ablation results to {out_file}")

    text_store.close()
    qdrant_store.close()


if __name__ == "__main__":
    main()
