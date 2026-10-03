"""PRISMX Sparse Vector Reindexing Module (PATCH-1)."""

from __future__ import annotations

import logging
from typing import Any
from qdrant_client import models

from prismx.config import load_config
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore, passage_id_to_point_id
from prismx.index.text_store import TextStore

logger = logging.getLogger("prismx.reindex")


def reindex_sparse_vectors(
    config: dict[str, Any] | None = None,
    threshold: float = 0.10,
    force: bool = False,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Recomputes avgdl_ref and updates Qdrant sparse vectors in batches."""
    if config is None:
        config = load_config()

    text_store = TextStore(db_path=config["storage"]["sqlite_path"])
    qdrant_store = QdrantStore(
        host=config["qdrant"]["host"],
        port=config["qdrant"]["port"],
        grpc_port=config["qdrant"]["grpc_port"],
        prefer_grpc=config["qdrant"].get("prefer_grpc", True),
        collection_name=config["qdrant"]["collection_name"],
    )
    tokenizer = BM25Tokenizer(k1=config["bm25"]["k1"], b=config["bm25"]["b"])

    stats = text_store.get_stats()
    n_docs = stats["n_docs"]
    total_len = stats["total_doc_len"]
    avgdl_ref = stats["avgdl_ref"]
    true_avgdl = stats["true_avgdl"]
    drift = stats["drift"]

    logger.info(f"Current stats: n_docs={n_docs}, total_tokens={total_len}")
    logger.info(f"Current avgdl_ref={avgdl_ref:.4f}, true_avgdl={true_avgdl:.4f}, drift={drift * 100:.2f}%")

    if drift <= threshold and not force:
        logger.info(f"Drift ({drift * 100:.2f}%) is within threshold ({threshold * 100:.2f}%). No reindexing required.")
        return {"status": "skipped", "drift": drift, "avgdl_ref": avgdl_ref}

    new_avgdl_ref = true_avgdl
    logger.info(f"Updating avgdl_ref from {avgdl_ref:.4f} to {new_avgdl_ref:.4f}...")
    text_store.set_meta("avgdl_ref", str(new_avgdl_ref))

    cursor = text_store.conn.cursor()
    cursor.execute("SELECT passage_id, text FROM passages")

    batch_points = []
    processed = 0
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for pid, text in rows:
            pt_id = passage_id_to_point_id(str(pid))
            s_indices, s_values = tokenizer.compute_doc_sparse_vector(text, new_avgdl_ref)
            batch_points.append(
                models.PointVectors(
                    id=pt_id,
                    vector={"bm25": models.SparseVector(indices=s_indices, values=s_values)},
                )
            )

        qdrant_store.client.update_vectors(
            collection_name=qdrant_store.collection_name,
            points=batch_points,
            wait=True,
        )
        processed += len(batch_points)
        batch_points = []
        if processed % 10000 == 0 or processed == n_docs:
            logger.info(f"Reindexed {processed}/{n_docs} documents...")

    text_store.close()
    qdrant_store.close()
    logger.info(f"Successfully reindexed {processed} documents with new avgdl_ref={new_avgdl_ref:.4f}.")
    return {"status": "success", "processed": processed, "new_avgdl_ref": new_avgdl_ref}
