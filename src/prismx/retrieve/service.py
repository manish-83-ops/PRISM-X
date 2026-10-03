"""PRISMX Unified Search and Storage Service implementing Gate 4A Reranking, Caching, and Evidence Telemetry."""

from __future__ import annotations

import logging
import time
from typing import Any

from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore, passage_id_to_point_id
from prismx.index.text_store import TextStore
from prismx.retrieve.cache import QueryCache
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.filters import build_qdrant_filter
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.rerank import CrossEncoderReranker
from prismx.schemas import (
    DeleteResponse,
    LatencyBreakdown,
    MetaResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    UpsertRequest,
    UpsertResponse,
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
        reranker: CrossEncoderReranker | None = None,
        cache: QueryCache | None = None,
    ):
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.text_store = text_store
        self.qdrant_store = qdrant_store
        self.config = config
        self.reranker = reranker
        self.cache = cache if cache is not None else QueryCache(maxsize=2000)
        self.inconsistency_count = 0

    def search(self, request: SearchRequest) -> SearchResponse:
        t_total_start = time.perf_counter()

        # Check Cache if enabled
        use_cache = request.get_use_cache()
        cache_key = None
        if use_cache and self.cache is not None:
            cache_key = self.cache.make_key(
                query=request.query,
                mode=request.mode,
                top_k=request.top_k,
                filters=request.filters.dict() if request.filters else None,
                fusion=request.fusion.dict() if request.fusion else None,
                rerank=request.rerank or request.mode in ("hybrid_rerank", "hybrid+rerank", "prismx"),
                rerank_k=request.rerank_k,
            )
            cached_resp: SearchResponse | None = self.cache.get(cache_key)
            if cached_resp is not None:
                # Return cached response with cache_hit=True and updated timing
                t_lookup_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)
                resp_copy = cached_resp.model_copy(deep=True)
                resp_copy.cache_hit = True
                resp_copy.latency_ms.total = t_lookup_ms
                return resp_copy

        query_filter = build_qdrant_filter(request.filters)

        latencies = {
            "encode": 0.0,
            "dense": 0.0,
            "sparse": 0.0,
            "fusion": 0.0,
            "fetch_text": 0.0,
            "rerank": 0.0,
            "total": 0.0,
        }

        should_rerank = (
            request.mode in ("hybrid_rerank", "hybrid+rerank", "prismx")
            or request.rerank
        ) and (self.reranker is not None)

        # 1. Retrieval Candidate Depth
        rerank_k = 10 if request.mode == "prismx" else (request.rerank_k if should_rerank else request.top_k)
        min_depth = self.config.get("retrieval", {}).get("candidate_depth", 40)
        candidate_depth = max(request.top_k * 4, rerank_k, min_depth)

        search_ef = getattr(request, "search_ef", None)

        if request.mode == "dense":
            candidates, enc_ms, dense_ms = self.dense_retriever.retrieve(
                query=request.query,
                limit=candidate_depth,
                query_filter=query_filter,
                search_ef=search_ef,
            )
            latencies["encode"] = round(enc_ms, 2)
            latencies["dense"] = round(dense_ms, 2)
            fused_candidates = candidates
            fusion_used = None
        else:
            # Hybrid or Hybrid+Rerank mode
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
                search_ef=search_ef,
            )
            latencies.update(h_lats)
            fusion_used = {"method": method, "alpha": alpha, "rrf_k": rrf_k}

        # 2. Text Fetch from SQLite implementing PATCH-2 (Ordered ID preservation & backfilling)
        t_fetch_start = time.perf_counter()
        ordered_pids = [c["passage_id"] for c in fused_candidates]
        hydrated_map = self.text_store.get_passages_by_ids(ordered_pids, chunk_size=400)
        latencies["fetch_text"] = round((time.perf_counter() - t_fetch_start) * 1000.0, 2)

        # 3. Assemble Candidate Pool with hydrated text
        hydrated_candidates: list[dict[str, Any]] = []
        for rank_idx, cand in enumerate(fused_candidates, start=1):
            pid = cand["passage_id"]
            if pid not in hydrated_map:
                self.inconsistency_count += 1
                logger.warning(f"Inconsistency: passage {pid} found in vector DB but missing in text store!")
                continue

            doc_info = hydrated_map[pid]
            retrieved_by = []
            if cand.get("dense_rank") is not None and cand.get("dense_rank", 0) > 0:
                retrieved_by.append("dense")
            if cand.get("bm25_rank") is not None and cand.get("bm25_rank", 0) > 0:
                retrieved_by.append("sparse")
            if not retrieved_by:
                retrieved_by = ["dense"] if request.mode == "dense" else ["hybrid"]

            c_entry = {
                "passage_id": pid,
                "text": doc_info["text"],
                "category": doc_info.get("category"),
                "source": doc_info.get("source"),
                "length_chars": len(doc_info["text"]),
                "score": cand.get("score") if request.mode != "dense" else cand.get("dense_score", 0.0),
                "dense_rank": cand.get("dense_rank"),
                "dense_score": cand.get("dense_score"),
                "bm25_rank": cand.get("bm25_rank"),
                "bm25_score": cand.get("bm25_score"),
                "sparse_rank": cand.get("bm25_rank"),
                "sparse_score": cand.get("bm25_score"),
                "fused_rank": rank_idx if request.mode != "dense" else None,
                "fused_score": cand.get("score") if request.mode != "dense" else None,
                "retrieved_by": retrieved_by,
                "rerank_rank": None,
                "rerank_score": None,
            }
            hydrated_candidates.append(c_entry)

        # 4. Optional Reranking Stage
        final_pool = hydrated_candidates
        governor_state = "normal"
        if should_rerank and self.reranker is not None:
            rerank_pool = hydrated_candidates[:rerank_k]
            rerank_budget = request.get_rerank_budget_ms()
            elapsed_pre_rerank = (time.perf_counter() - t_total_start) * 1000.0
            total_deadline = getattr(request, "total_deadline_ms", 250.0) or 250.0
            # Gate 5.2 (1f): Cross-encoder uses min(rerank_budget_ms, total_deadline_ms - elapsed - 10ms safety)
            remaining_for_rerank = total_deadline - elapsed_pre_rerank - 10.0
            effective_deadline = min(rerank_budget, max(0.0, remaining_for_rerank))

            t_rerank_stage_start = time.perf_counter()
            reranked, r_ms, gov_state = self.reranker.rerank(
                query=request.query,
                candidates=rerank_pool,
                top_k=request.top_k,
                max_length=128,
                deadline_ms=effective_deadline,
                t_request_start=t_rerank_stage_start,
                batch_size=5,
            )
            latencies["rerank"] = round(r_ms, 2)
            final_pool = reranked
            governor_state = gov_state

        # 5. Assemble final response items
        results: list[SearchResultItem] = []
        for rank_idx, item in enumerate(final_pool[: request.top_k], start=1):
            final_score = (
                item.get("rerank_score")
                if should_rerank and item.get("rerank_score") is not None
                else item["score"]
            )
            results.append(
                SearchResultItem(
                    rank=rank_idx,
                    passage_id=item["passage_id"],
                    text=item["text"],
                    category=item.get("category"),
                    source=item.get("source"),
                    length_chars=item.get("length_chars"),
                    score=round(float(final_score), 4),
                    dense_rank=item.get("dense_rank"),
                    dense_score=item.get("dense_score"),
                    bm25_rank=item.get("bm25_rank"),
                    bm25_score=item.get("bm25_score"),
                    sparse_rank=item.get("sparse_rank"),
                    sparse_score=item.get("sparse_score"),
                    fused_rank=item.get("fused_rank"),
                    fused_score=item.get("fused_score"),
                    rerank_rank=item.get("rerank_rank"),
                    rerank_score=item.get("rerank_score"),
                    retrieved_by=item.get("retrieved_by", []),
                )
            )

        latencies["total"] = round((time.perf_counter() - t_total_start) * 1000.0, 2)
        index_version = int(self.text_store.get_meta("index_version", 1))

        filters_applied = None
        if request.filters:
            filters_applied = {
                k: v
                for k, v in [("category", request.filters.category), ("source", request.filters.source)]
                if v is not None
            }

        response = SearchResponse(
            query=request.query,
            mode=request.mode,
            fusion_used=fusion_used,
            filters_applied=filters_applied,
            index_version=index_version,
            results=results,
            latency_ms=LatencyBreakdown(**latencies),
            cache_hit=False,
            governor_state=governor_state,
        )

        # Store in cache if enabled
        if use_cache and self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, response)

        return response

    def upsert_passage(self, req: UpsertRequest) -> UpsertResponse:
        """Live upsert implementing PATCH-1 & Gate 6 atomic update with cache invalidation."""
        pid_str = str(req.passage_id)
        pt_id = passage_id_to_point_id(pid_str)

        # Invalidate query cache immediately
        if self.cache is not None:
            self.cache.invalidate()

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

        sparse_name = "sparse"
        try:
            c_info = self.qdrant_store.client.get_collection(self.qdrant_store.collection_name)
            s_vecs = c_info.config.params.sparse_vectors or {}
            sparse_name = "bm25" if "bm25" in s_vecs else "sparse"
        except Exception:
            sparse_name = "sparse"

        pt = models.PointStruct(
            id=pt_id,
            vector={
                "dense": dense_vec,
                sparse_name: models.SparseVector(indices=sparse_indices, values=sparse_values),
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
        """Live delete removing from Qdrant and SQLite with cache invalidation."""
        pid_str = str(passage_id)

        # Invalidate query cache immediately
        if self.cache is not None:
            self.cache.invalidate()

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
        cache_stats = self.cache.stats() if self.cache is not None else None
        return {
            "modes": ["dense", "hybrid", "hybrid_rerank"],
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
                "reranker": (
                    self.reranker.model_name
                    if self.reranker is not None
                    else "cross-encoder/ms-marco-MiniLM-L-6-v2 (INT8)"
                ),
            },
            "cache_stats": cache_stats,
        }
