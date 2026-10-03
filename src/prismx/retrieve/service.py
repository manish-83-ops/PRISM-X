"""PRISMX Unified Search and Storage Service implementing PATCH-2 Decoupled Ordering."""

from __future__ import annotations

import logging
import time
from typing import Any

from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.text_store import TextStore
from prismx.index.qdrant_store import QdrantStore, passage_id_to_point_id
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.filters import build_qdrant_filter
from prismx.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    LatencyBreakdown,
    UpsertRequest,
    UpsertResponse,
    DeleteResponse,
    MetaResponse,
)

logger = logging.getLogger("prismx.service")

class SearchService:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        text_store: TextStore,
        qdrant_store: QdrantStore,
        config: dict[str, Any],
    ):
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.text_store = text_store
        self.qdrant_store = qdrant_store
        self.config = config
        self.inconsistency_count = 0

    def search(self, request: SearchRequest) -> SearchResponse:
        t_total_start = time.perf_counter()
        query_filter = build_qdrant_filter(request.filters)

        latencies = {
            "encode": 0.0,
            "dense": 0.0,
            "sparse": 0.0,
            "fusion": 0.0,
            "fetch_text": 0.0,
            "total": 0.0,
        }

        # 1. Retrieval
        candidate_depth = max(request.top_k * 4, self.config["retrieval"]["candidate_depth"])

        if request.mode == "dense":
            candidates, enc_ms, dense_ms = self.dense_retriever.retrieve(
                query=request.query,
                limit=candidate_depth,
                query_filter=query_filter,
            )
            latencies["encode"] = round(enc_ms, 2)
            latencies["dense"] = round(dense_ms, 2)
            fused_candidates = candidates
            fusion_used = None
        else:
            # Hybrid mode
            f_params = request.fusion
            method = f_params.method if f_params else self.config["retrieval"]["fusion"]["method"]
            alpha = f_params.alpha if f_params else self.config["retrieval"]["fusion"]["alpha"]
            rrf_k = f_params.rrf_k if f_params else self.config["retrieval"]["fusion"]["rrf_k"]

            fused_candidates, h_lats = self.hybrid_retriever.retrieve(
                query=request.query,
                limit=candidate_depth,
                query_filter=query_filter,
                fusion_method=method,
                alpha=alpha,
                rrf_k=rrf_k,
            )
            latencies.update(h_lats)
            fusion_used = {"method": method, "alpha": alpha, "rrf_k": rrf_k}

        # 2. Text Fetch from SQLite implementing PATCH-2 (Ordered ID preservation & backfilling)
        t_fetch_start = time.perf_counter()
        ordered_pids = [c["passage_id"] for c in fused_candidates]
        hydrated_map = self.text_store.get_passages_by_ids(ordered_pids, chunk_size=400)
        latencies["fetch_text"] = round((time.perf_counter() - t_fetch_start) * 1000.0, 2)

        # 3. Assemble final results strictly by candidate ordering
        results: list[SearchResultItem] = []
        for cand in fused_candidates:
            pid = cand["passage_id"]
            if pid not in hydrated_map:
                # Inconsistency detected: ID in Qdrant but missing from SQLite
                self.inconsistency_count += 1
                logger.warning(f"Inconsistency: passage {pid} found in vector DB but missing in text store!")
                continue

            doc_info = hydrated_map[pid]
            results.append(
                SearchResultItem(
                    rank=len(results) + 1,
                    passage_id=pid,
                    text=doc_info["text"],
                    category=doc_info.get("category"),
                    source=doc_info.get("source"),
                    score=cand.get("score") if request.mode == "hybrid" else cand.get("dense_score", 0.0),
                    dense_rank=cand.get("dense_rank"),
                    dense_score=cand.get("dense_score"),
                    bm25_rank=cand.get("bm25_rank"),
                    bm25_score=cand.get("bm25_score"),
                )
            )

            if len(results) == request.top_k:
                break

        latencies["total"] = round((time.perf_counter() - t_total_start) * 1000.0, 2)
        index_version = int(self.text_store.get_meta("index_version", 1))

        filters_applied = None
        if request.filters:
            filters_applied = {
                k: v for k, v in [("category", request.filters.category), ("source", request.filters.source)]
                if v is not None
            }

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            fusion_used=fusion_used,
            filters_applied=filters_applied,
            index_version=index_version,
            results=results,
            latency_ms=LatencyBreakdown(**latencies),
        )

    def upsert_passage(self, req: UpsertRequest) -> UpsertResponse:
        """Live upsert implementing PATCH-1 & Gate 6 atomic update."""
        pid_str = str(req.passage_id)
        pt_id = passage_id_to_point_id(pid_str)

        # 1. Compute dense vector
        dense_vec = self.dense_retriever.encoder.encode_queries(req.text)[0].tolist()

        # 2. Compute BM25 sparse vector using frozen avgdl_ref
        avgdl_ref = float(self.text_store.get_meta("avgdl_ref", 50.0))
        sparse_indices, sparse_values = self.hybrid_retriever.tokenizer.compute_doc_sparse_vector(req.text, avgdl_ref)

        # 3. Derive category if not provided
        category = req.category or "general"

        # 4. Write to SQLite (updates running total_doc_len, n_docs, index_version)
        tokens = self.hybrid_retriever.tokenizer.tokenize(req.text)
        new_version = self.text_store.upsert_single(
            passage_id=pid_str,
            text=req.text,
            category=category,
            source=req.source or "manual",
            doc_token_len=len(tokens),
        )

        # 5. Write to Qdrant (wait=True for synchronous confirmation)
        from qdrant_client import models
        pt = models.PointStruct(
            id=pt_id,
            vector={
                "dense": dense_vec,
                "bm25": models.SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload={
                "passage_id": pid_str,
                "category": category,
                "source": req.source or "manual",
            },
        )
        self.qdrant_store.upsert_points_batch([pt], wait=True)

        return UpsertResponse(
            status="success",
            passage_id=pid_str,
            index_version=new_version,
        )

    def delete_passage(self, passage_id: str) -> DeleteResponse:
        """Live delete removing from Qdrant and SQLite."""
        pid_str = str(passage_id)
        was_deleted, new_version = self.text_store.delete_single(pid_str)
        self.qdrant_store.delete_point(pid_str, wait=True)

        return DeleteResponse(
            status="deleted" if was_deleted else "not_found",
            passage_id=pid_str,
            index_version=new_version,
        )

    def get_meta(self) -> dict[str, Any]:
        """Provides system health, drift, index stats, and metadata."""
        stats = self.text_store.get_stats()
        q_count = self.qdrant_store.count()
        return {
            "modes": ["dense", "hybrid"],
            "fusion_defaults": self.config["retrieval"]["fusion"],
            "point_count": q_count,
            "sqlite_count": stats["n_docs"],
            "index_version": stats["index_version"],
            "avgdl_ref": stats["avgdl_ref"],
            "true_avgdl": stats["true_avgdl"],
            "drift": stats["drift"],
            "drift_warning": stats["drift_exceeds_threshold"],
            "inconsistency_count": self.inconsistency_count,
            "config_hash": self.config.get("_config_hash", "unknown"),
            "models": {
                "dense": self.config["encoder"]["model_name"],
                "lexical": "qdrant_sparse_bm25_idf",
            },
        }
