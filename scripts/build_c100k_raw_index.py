"""Build Qdrant vector index and SQLite text store for c100k_raw.
In strict accordance with ADR-016 (B5):
- BGE-small dense embeddings (dim=384, normalize=True)
- Qdrant HNSW vector index (M=16, ef_construct=100)
- Sparse BM25 with dynamic IDF modifier
- SQLite text hydration store
- Logs indexing time, RAM peak, index size, duplicate stats, positive-to-total ratio, per-query_type counts
"""

import gc
import json
import os
import psutil
import time
from pathlib import Path
import numpy as np
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from prismx.index.lexical import BM25Tokenizer, stable_token_hash
from prismx.index.text_store import TextStore


COLLECTION_NAME = "c100k_raw"
PASSAGES_PATH = Path("data/c100k_raw/c100k_raw_passages.jsonl")
DB_PATH = Path("data/c100k_raw/text_store_raw.db")
STATS_PATH = Path("data/c100k_raw/dataset_stats.json")
LOG_PATH = Path("results/c100k_raw/build_stats.json")


def get_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def build_c100k_raw_index(batch_size: int = 128, torch_threads: int = 8):
    t_start = time.time()
    ram_peak = get_ram_mb()

    out_results = Path("results/c100k_raw")
    out_results.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("STARTING C100K_RAW INDEX BUILD (ADR-016 / B5)")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Passages Source: {PASSAGES_PATH}")
    print("=================================================================")

    # 1. Load Passages
    print("\n[Step 1/5] Loading passages from disk...")
    passages = []
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "source" not in item:
                item["source"] = item.get("url", "msmarco")
            passages.append(item)
    n_passages = len(passages)
    print(f"Loaded {n_passages:,} passages.")
    ram_peak = max(ram_peak, get_ram_mb())

    # 2. Ingest SQLite Text Store
    print(f"\n[Step 2/5] Ingesting into SQLite text store ({DB_PATH})...")
    t_sql_start = time.time()
    if DB_PATH.exists():
        DB_PATH.unlink()
    text_store = TextStore(db_path=str(DB_PATH))
    text_store.insert_passages_bulk(passages, batch_size=5000)
    text_store.set_meta("index_version", 1)
    text_store.set_meta("collection_name", COLLECTION_NAME)
    sql_time = time.time() - t_sql_start
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"SQLite ingestion finished in {sql_time:.2f}s (DB size: {db_size_mb:.2f} MB).")
    ram_peak = max(ram_peak, get_ram_mb())

    # 3. Initialize Qdrant Collection
    print(f"\n[Step 3/5] Setting up Qdrant collection '{COLLECTION_NAME}'...")
    client = QdrantClient(host="127.0.0.1", port=6333)
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists. Recreating...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    # Create payload index for category and source
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print("Qdrant collection and payload indexes created.")

    # 4. Tokenize BM25 Sparse
    print("\n[Step 4/5] Computing sparse BM25 representations...")
    t_sparse_start = time.time()
    tokenizer = BM25Tokenizer()
    sparse_vectors = []
    for p in passages:
        toks = tokenizer.tokenize(p["text"])
        tf_counts = {}
        for t in toks:
            h = stable_token_hash(t)
            tf_counts[h] = tf_counts.get(h, 0) + 1
        indices = list(tf_counts.keys())
        values = [float(v) for v in tf_counts.values()]
        sparse_vectors.append(models.SparseVector(indices=indices, values=values))
    sparse_time = time.time() - t_sparse_start
    print(f"Sparse vectors computed in {sparse_time:.2f}s.")
    ram_peak = max(ram_peak, get_ram_mb())

    # 5. Dense Encoding & Qdrant Upsert
    print(f"\n[Step 5/5] Encoding {n_passages:,} passages with BGE-small (threads={torch_threads}, batch={batch_size})...")
    torch.set_num_threads(torch_threads)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    
    t_enc_start = time.time()
    total_batches = (n_passages + batch_size - 1) // batch_size
    
    upsert_chunk_size = 1000
    points_buffer = []

    for b_idx in range(total_batches):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, n_passages)
        batch_passages = passages[start_i:end_i]
        batch_texts = [p["text"] for p in batch_passages]

        embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            show_progress_bar=False
        )

        for p_rel, (p, emb) in enumerate(zip(batch_passages, embeddings)):
            global_idx = start_i + p_rel
            points_buffer.append(
                models.PointStruct(
                    id=global_idx + 1,
                    vector={
                        "dense": emb.tolist(),
                        "sparse": sparse_vectors[global_idx]
                    },
                    payload={
                        "passage_id": p["passage_id"],
                        "category": p["category"],
                        "source": p["url"],
                        "length_chars": len(p["text"]),
                        "originating_query_ids": p.get("originating_query_ids", [])
                    }
                )
            )

        if len(points_buffer) >= upsert_chunk_size or b_idx == total_batches - 1:
            client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
            points_buffer.clear()

        ram_peak = max(ram_peak, get_ram_mb())
        if (b_idx + 1) % 50 == 0 or b_idx == total_batches - 1:
            elapsed_enc = time.time() - t_enc_start
            rate = (end_i) / elapsed_enc
            remaining_s = (n_passages - end_i) / rate if rate > 0 else 0
            print(f"Batch {b_idx + 1}/{total_batches} ({end_i:,}/{n_passages:,} passages) | Rate: {rate:.1f} pass/s | ETA: {remaining_s/60:.1f}m | RAM: {get_ram_mb():.1f} MB")

    total_time = time.time() - t_start
    c_info = client.get_collection(COLLECTION_NAME)

    # Read dataset stats
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        d_stats = json.load(f)

    import platform
    import transformers
    import sentence_transformers
    import importlib.metadata

    lib_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "qdrant_client": importlib.metadata.version("qdrant-client"),
        "numpy": np.__version__,
        "psutil": psutil.__version__,
    }

    build_report = {
        "collection_name": COLLECTION_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points_count": c_info.points_count,
        "sqlite_count": n_passages,
        "total_indexing_time_seconds": round(total_time, 2),
        "total_indexing_time_minutes": round(total_time / 60.0, 2),
        "sqlite_db_size_mb": round(db_size_mb, 2),
        "ram_peak_mb": round(ram_peak, 2),
        "library_versions": lib_versions,
        "stage_timings": {
            "sqlite_s": round(sql_time, 2),
            "sparse_bm25_s": round(sparse_time, 2),
            "dense_encode_and_upsert_s": round(time.time() - t_enc_start, 2)
        },
        "dataset_stats": d_stats
    }

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(build_report, f, indent=2)

    print("\n=================================================================")
    print("C100K_RAW INDEX BUILD COMPLETE!")
    print(f"Indexed: {c_info.points_count:,} points in Qdrant")
    print(f"SQLite DB Size: {db_size_mb:.2f} MB")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"RAM Peak: {ram_peak:.1f} MB")
    print("Library Versions:")
    for k, v in lib_versions.items():
        print(f"  {k}: {v}")
    print("=================================================================")
    return build_report



if __name__ == "__main__":
    build_c100k_raw_index()
