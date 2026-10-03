"""Candidate Movement Analysis: Hybrid vs Dense and BM25 vs Dense on BENCH 100 queries."""

import json
from pathlib import Path
from prismx.config import load_config
from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    cfg = load_config()
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
        torch_threads=8,
    )
    tokenizer = BM25Tokenizer(
        k1=cfg["lexical"]["k1"],
        b=cfg["lexical"]["b"],
        use_stemming=cfg["lexical"].get("stemming", False),
    )
    dense_retriever = DenseRetriever(encoder=encoder, qdrant_store=qdrant_store, config=cfg)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        tokenizer=tokenizer,
        qdrant_store=qdrant_store,
        config=cfg,
    )

    with open(REPO_ROOT / "data" / "manifests" / "split_bench.json", "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    # Counters
    # 1. Hybrid vs Dense
    hybrid_added = {10: 0, 20: 0, 50: 0}
    hybrid_removed = {10: 0, 20: 0, 50: 0}

    # 2. BM25 standalone vs Dense standalone
    bm25_has_dense_misses = {10: 0, 20: 0, 50: 0}
    dense_has_bm25_misses = {10: 0, 20: 0, 50: 0}

    # Tracking exact query details
    detail_records = []

    for item in bench_queries:
        qid = str(item["query_id"])
        query = item["query"]
        gold_ids = set(str(g) for g in item["gold_passage_ids"])

        # Dense candidates (limit=50)
        dense_cands, _, _ = dense_retriever.retrieve(query, limit=50)
        dense_pids = [str(c["passage_id"]) for c in dense_cands]

        # BM25 candidates (limit=50)
        from qdrant_client import models
        sparse_indices, sparse_values = tokenizer.compute_query_sparse_vector(query)
        sparse_pids = []
        if sparse_indices:
            resp = qdrant_store.client.query_points(
                collection_name=qdrant_store.collection_name,
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="bm25",
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            sparse_pids = [str(p.payload.get("passage_id", p.id)) for p in resp.points]

        # Hybrid fused candidates (limit=50)
        fused_cands, _ = hybrid_retriever.retrieve(query, limit=50, alpha=0.8, norm_method="minmax")
        hybrid_pids = [str(c["passage_id"]) for c in fused_cands]

        record = {"qid": qid, "gold": list(gold_ids)}

        for K in [10, 20, 50]:
            d_k = set(dense_pids[:K])
            s_k = set(sparse_pids[:K])
            h_k = set(hybrid_pids[:K])

            d_hits = gold_ids.intersection(d_k)
            s_hits = gold_ids.intersection(s_k)
            h_hits = gold_ids.intersection(h_k)

            # Hybrid vs Dense
            if len(h_hits - d_hits) > 0:
                hybrid_added[K] += 1
            if len(d_hits - h_hits) > 0:
                hybrid_removed[K] += 1

            # BM25 standalone vs Dense
            if len(s_hits - d_hits) > 0:
                bm25_has_dense_misses[K] += 1
            if len(d_hits - s_hits) > 0:
                dense_has_bm25_misses[K] += 1

            record[f"dense_hits_{K}"] = len(d_hits)
            record[f"sparse_hits_{K}"] = len(s_hits)
            record[f"hybrid_hits_{K}"] = len(h_hits)

        detail_records.append(record)

    print("====================================================================")
    print("BENCH QUERY CANDIDATE ANALYSIS (N = 100 QUERIES)")
    print("====================================================================")
    print("\n--- 1. Hybrid (Dense + BM25 alpha=0.8) vs Dense Baseline ---")
    for K in [10, 20, 50]:
        print(f"Top-{K}: Hybrid added gold in {hybrid_added[K]} queries, removed gold in {hybrid_removed[K]} queries.")

    print("\n--- 2. BM25 Standalone vs Dense Baseline ---")
    for K in [10, 20, 50]:
        print(f"Top-{K}: BM25 contained gold that Dense missed in {bm25_has_dense_misses[K]} queries.")
        print(f"Top-{K}: Dense contained gold that BM25 missed in {dense_has_bm25_misses[K]} queries.")

    out_path = REPO_ROOT / "results" / "candidate_movement_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "hybrid_vs_dense": {
                "added": hybrid_added,
                "removed": hybrid_removed,
            },
            "bm25_vs_dense": {
                "bm25_unique_golds": bm25_has_dense_misses,
                "dense_unique_golds": dense_has_bm25_misses,
            },
            "details": detail_records,
        }, f, indent=2)
    print(f"\nSaved full candidate analysis to {out_path}")

if __name__ == "__main__":
    main()
