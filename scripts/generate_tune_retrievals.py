"""Dump per-query retrievals for TUNE (500 queries) across Dense, Sparse, Hybrid, and Rerank."""

import json
import os
import sys
import time
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore
from prismx.index.text_store import TextStore
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.rerank import CrossEncoderReranker
from qdrant_client import models

def main():
    tune_path = REPO_ROOT / "data" / "c100k_raw" / "tune_raw_500.json"
    out_path = REPO_ROOT / "results" / "tune_per_query_retrievals.json"
    
    with open(tune_path, "r", encoding="utf-8") as f:
        tune_queries = json.load(f)

    print(f"Loaded {len(tune_queries)} TUNE queries.")

    qdrant = QdrantStore(host="127.0.0.1", port=6333, collection_name="c100k_raw")
    encoder = DenseEncoder("BAAI/bge-small-en-v1.5", embedding_dim=384, max_seq_length=128, torch_threads=8)
    tokenizer = BM25Tokenizer()
    text_store = TextStore(db_path=str(REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"))
    dense_retriever = DenseRetriever(encoder=encoder, qdrant_store=qdrant, default_candidate_depth=50)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        tokenizer=tokenizer,
        qdrant_store=qdrant,
        default_candidate_depth=50,
        default_alpha=0.8,
    )
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", torch_threads=8)

    results = []
    t0 = time.time()
    for idx, q in enumerate(tune_queries):
        qid = str(q["query_id"])
        query = q["query"]
        gold_pids = [str(g) for g in q["gold_pids"]]

        # 1. Dense (limit=50)
        dense_cands, _, _ = dense_retriever.retrieve(query, limit=50, search_ef=64)
        dense_pids = [str(c["passage_id"]) for c in dense_cands]

        # 2. Sparse (BM25 standalone, limit=50)
        sparse_indices, sparse_values = tokenizer.compute_query_sparse_vector(query)
        sparse_pids = []
        if sparse_indices:
            resp = qdrant.client.query_points(
                collection_name="c100k_raw",
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            sparse_pids = [str(p.payload.get("passage_id", p.id)) for p in resp.points]

        # 3. Hybrid (alpha=0.8, limit=50)
        fused_cands, _ = hybrid_retriever.retrieve(query, limit=50, alpha=0.8, norm_method="minmax", search_ef=64)
        hybrid_pids = [str(c["passage_id"]) for c in fused_cands]

        # 4. Rerank (top 10 hydrated)
        top10_cands = fused_cands[:10]
        hydrated = text_store.get_passages_by_ids([c["passage_id"] for c in top10_cands])
        rerank_input = [{
            "passage_id": c["passage_id"],
            "text": hydrated.get(c["passage_id"], {}).get("text", ""),
            "score": c["score"]
        } for c in top10_cands]

        reranked, _, _ = reranker.rerank(query, rerank_input, top_k=10, max_length=128, deadline_ms=200.0)
        rerank_pids = [str(c["passage_id"]) for c in reranked] + hybrid_pids[10:]

        results.append({
            "query_id": qid,
            "query": query,
            "gold_pids": gold_pids,
            "dense_pids": dense_pids,
            "sparse_pids": sparse_pids,
            "hybrid_pids": hybrid_pids,
            "rerank_pids": rerank_pids,
        })

        if (idx + 1) % 50 == 0:
            print(f"[{idx + 1}/500] Processed ({time.time() - t0:.1f}s)")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} per-query TUNE retrievals to {out_path} ({time.time() - t0:.1f}s)")

if __name__ == "__main__":
    main()
