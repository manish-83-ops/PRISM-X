"""PRISMX Dense Retrieval Channel."""

from __future__ import annotations

import time
from typing import Any
from qdrant_client import models
from prismx.index.encoder import DenseEncoder
from prismx.index.qdrant_store import QdrantStore

class DenseRetriever:
    def __init__(
        self,
        encoder: DenseEncoder,
        qdrant_store: QdrantStore,
        default_candidate_depth: int = 50,
        config: dict[str, Any] | None = None,
    ):
        self.encoder = encoder
        self.qdrant_store = qdrant_store
        self.config = config or {}
        if "retrieval" in self.config:
            self.default_candidate_depth = self.config["retrieval"].get("candidate_depth", default_candidate_depth)
        else:
            self.default_candidate_depth = default_candidate_depth

        self.search_ef = None
        if "qdrant" in self.config:
            self.search_ef = self.config["qdrant"].get("search_ef")

    def retrieve(
        self,
        query: str,
        limit: int | None = None,
        query_filter: models.Filter | None = None,
        prefix: str = "",
        search_ef: int | None = None,
    ) -> tuple[list[dict[str, Any]], float, float]:
        """Executes dense vector search against Qdrant HNSW index.
        
        Returns:
            candidates: list of dicts with passage_id, score, dense_rank
            encode_time_ms: latency of query embedding in ms
            search_time_ms: latency of Qdrant HNSW search in ms
        """
        k = limit if limit is not None else self.default_candidate_depth
        ef_val = search_ef if search_ef is not None else self.search_ef
        search_params = models.SearchParams(hnsw_ef=ef_val) if ef_val is not None else None

        t0 = time.perf_counter()
        q_emb = self.encoder.encode_queries(query, prefix=prefix)[0]
        encode_time_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        response = self.qdrant_store.client.query_points(
            collection_name=self.qdrant_store.collection_name,
            query=q_emb.tolist(),
            using="dense",
            query_filter=query_filter,
            search_params=search_params,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        search_time_ms = (time.perf_counter() - t1) * 1000.0

        candidates = []
        for rank, pt in enumerate(response.points, start=1):
            candidates.append({
                "passage_id": str(pt.payload.get("passage_id", pt.id)),
                "dense_score": float(pt.score),
                "dense_rank": rank,
                "category": pt.payload.get("category"),
                "source": pt.payload.get("source"),
            })

        return candidates, encode_time_ms, search_time_ms
