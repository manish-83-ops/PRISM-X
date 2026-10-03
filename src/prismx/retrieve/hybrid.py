"""PRISMX Hybrid Retrieval Channel."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any
from qdrant_client import models

from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.fusion import FusionEngine, NormalizationType

class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        tokenizer: BM25Tokenizer,
        qdrant_store: QdrantStore,
        default_candidate_depth: int = 50,
        default_alpha: float = 0.7,
        default_rrf_k: int = 60,
        default_fusion_method: str = "weighted",
        config: dict[str, Any] | None = None,
    ):
        self.dense_retriever = dense_retriever
        self.tokenizer = tokenizer
        self.qdrant_store = qdrant_store
        self.config = config or {}
        if "retrieval" in self.config:
            self.default_candidate_depth = self.config["retrieval"].get("candidate_depth", default_candidate_depth)
            f_cfg = self.config["retrieval"].get("fusion", {})
            self.default_alpha = f_cfg.get("alpha", default_alpha)
            self.default_rrf_k = f_cfg.get("rrf_k", default_rrf_k)
            self.default_fusion_method = f_cfg.get("method", default_fusion_method)
        else:
            self.default_candidate_depth = default_candidate_depth
            self.default_alpha = default_alpha
            self.default_rrf_k = default_rrf_k
            self.default_fusion_method = default_fusion_method

    def _retrieve_sparse(
        self,
        query: str,
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        sparse_indices, sparse_values = self.tokenizer.compute_query_sparse_vector(query)
        sparse_cands = []

        if sparse_indices:
            # Auto-detect sparse vector name from collection params ('bm25' or 'sparse')
            sparse_name = getattr(self, "_sparse_vector_name", None)
            if sparse_name is None:
                try:
                    c_info = self.qdrant_store.client.get_collection(self.qdrant_store.collection_name)
                    s_vecs = c_info.config.params.sparse_vectors or {}
                    sparse_name = "bm25" if "bm25" in s_vecs else ("sparse" if "sparse" in s_vecs else "bm25")
                except Exception:
                    sparse_name = "bm25"
                self._sparse_vector_name = sparse_name

            resp = self.qdrant_store.client.query_points(
                collection_name=self.qdrant_store.collection_name,
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using=sparse_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            for rank, pt in enumerate(resp.points, start=1):
                sparse_cands.append({
                    "passage_id": str(pt.payload.get("passage_id", pt.id)),
                    "bm25_score": float(pt.score),
                    "bm25_rank": rank,
                    "category": pt.payload.get("category"),
                    "source": pt.payload.get("source"),
                })

        sparse_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return sparse_cands, sparse_ms

    def retrieve(
        self,
        query: str,
        limit: int | None = None,
        query_filter: models.Filter | None = None,
        fusion_method: str | None = None,
        alpha: float | None = None,
        rrf_k: int | None = None,
        norm_method: NormalizationType = "minmax",
        prefix: str = "",
        search_ef: int | None = None,
        concurrent_execution: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Executes dual-channel retrieval (dense + BM25 sparse) with client-side fusion.
        When concurrent_execution=True, executes dense encode+search concurrently with sparse BM25 search.
        
        Returns:
            fused_candidates: list of fused candidate dictionaries
            stage_latencies_ms: breakdown of encode, dense, sparse, and fusion latencies
        """
        k = limit if limit is not None else self.default_candidate_depth
        method = fusion_method or self.default_fusion_method
        a = alpha if alpha is not None else self.default_alpha
        rk = rrf_k if rrf_k is not None else self.default_rrf_k

        latencies: dict[str, float] = {}

        if concurrent_execution:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                dense_fut = executor.submit(
                    self.dense_retriever.retrieve,
                    query=query,
                    limit=k,
                    query_filter=query_filter,
                    prefix=prefix,
                    search_ef=search_ef,
                )
                sparse_fut = executor.submit(
                    self._retrieve_sparse,
                    query=query,
                    limit=k,
                    query_filter=query_filter,
                )
                dense_cands, encode_ms, dense_ms = dense_fut.result()
                sparse_cands, sparse_ms = sparse_fut.result()
        else:
            dense_cands, encode_ms, dense_ms = self.dense_retriever.retrieve(
                query=query,
                limit=k,
                query_filter=query_filter,
                prefix=prefix,
                search_ef=search_ef,
            )
            sparse_cands, sparse_ms = self._retrieve_sparse(
                query=query,
                limit=k,
                query_filter=query_filter,
            )

        latencies["encode"] = round(encode_ms, 2)
        latencies["dense"] = round(dense_ms, 2)
        latencies["sparse"] = round(sparse_ms, 2)

        # 3. Client-side fusion
        t_f0 = time.perf_counter()
        if method == "rrf":
            fused = FusionEngine.fuse_rrf(dense_cands, sparse_cands, rrf_k=rk)
        else:
            fused = FusionEngine.fuse_weighted(dense_cands, sparse_cands, alpha=a, norm_method=norm_method)

        latencies["fusion"] = round((time.perf_counter() - t_f0) * 1000.0, 2)
        return fused, latencies
