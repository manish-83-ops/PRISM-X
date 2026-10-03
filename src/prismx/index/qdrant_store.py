"""PRISMX Qdrant Store Management with Dual Vectors and Keyword Payload Indexing."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Sequence
from qdrant_client import QdrantClient, models

logger = logging.getLogger("prismx.qdrant_store")

DEFAULT_COLLECTION = "prismx_corpus"

def passage_id_to_point_id(passage_id: str) -> int:
    """Deterministically maps a passage ID string to a positive 64-bit integer point ID."""
    p_str = str(passage_id).strip()
    if p_str.isdigit():
        return int(p_str)
    if p_str.startswith("raw_") and p_str[4:].isdigit():
        return int(p_str[4:])
    # Fallback to stable 60-bit integer from SHA-256
    digest = hashlib.sha256(p_str.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)

class QdrantStore:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=f"http://{host}:{port}",
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            timeout=60,
        )

    def init_collection(
        self,
        embedding_dim: int = 384,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        recreate: bool = False,
    ) -> None:
        """Creates collection with dense vector + sparse BM25 vector (with IDF modifier) and payload indexes."""
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            print(f"Deleting existing collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            print(f"Creating Qdrant collection '{self.collection_name}' with dense (dim={embedding_dim}) and sparse (Modifier.IDF)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(
                            m=hnsw_m,
                            ef_construct=hnsw_ef_construct,
                        ),
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )

            # Create keyword payload indexes for pre-retrieval filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            print("Collection and keyword payload indexes initialized.")

    def upsert_points_batch(
        self,
        points: Sequence[models.PointStruct],
        wait: bool = True,
    ) -> None:
        """Upserts a batch of PointStructs with synchronous wait semantics."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=wait,
        )

    def delete_point(self, passage_id: str, wait: bool = True) -> None:
        """Deletes a single point by passage ID."""
        pt_id = passage_id_to_point_id(passage_id)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[pt_id],
            wait=wait,
        )

    def count(self) -> int:
        """Returns total points in collection."""
        res = self.client.count(collection_name=self.collection_name, exact=True)
        return res.count

    def sample_ids(self, limit: int = 200) -> list[str]:
        """Scrolls up to limit points and returns passage_ids."""
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return [str(pt.payload.get("passage_id", pt.id)) for pt in points]
        except Exception as exc:
            logger.warning(f"Error sampling points from Qdrant: {exc}")
            return []

    def close(self) -> None:
        """Closes underlying client connections."""
        try:
            self.client.close()
        except Exception:
            pass
