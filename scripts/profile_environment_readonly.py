"""Read-Only Environment & Serving Profile Script (Gate 6 Step 2).
Gathers all static environment, runtime, Qdrant, SQLite, and filesystem properties without timing benchmarks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import requests
import torch

def inspect_profile():
    print("=" * 60)
    print("GATE 6 STEP 2: READ-ONLY SERVING & ENVIRONMENT PROFILE")
    print("=" * 60)

    # 1. PyTorch & Threading
    print("\n[1. PyTorch & Threading]")
    print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '<not set>')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', '<not set>')}")
    print(f"  TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', '<not set>')}")

    # 2. TensorFlow status
    print("\n[2. TensorFlow Status]")
    try:
        import tensorflow as tf
        print(f"  TensorFlow imported: YES (version {tf.__version__})")
    except ImportError:
        print("  TensorFlow imported: NO")

    # 3. Model & Encoder Config
    print("\n[3. Model & Encoder Config]")
    from prismx.index.encoder import DenseEncoder
    encoder = DenseEncoder(model_name="BAAI/bge-small-en-v1.5", embedding_dim=384, max_seq_length=128, torch_threads=8)
    print(f"  DenseEncoder max_seq_length: {encoder.max_seq_length}")
    print(f"  DenseEncoder device: {encoder.device}")

    print("\n[4. Qdrant Protocol & Call Patterns]")
    from prismx.index.qdrant_store import QdrantStore
    qstore = QdrantStore(host="127.0.0.1", port=6333, collection_name="c100k_raw")
    print(f"  Client target: 127.0.0.1:6333 (collection: {qstore.collection_name})")
    # Inspect Qdrant client object
    client_type = type(qstore.client).__name__
    print(f"  Qdrant client class: {client_type}")
    
    # Check collection status via HTTP REST API
    try:
        resp = requests.get("http://127.0.0.1:6333/collections/c100k_raw").json()
        result = resp.get("result", {})
        status = result.get("status")
        opt_status = result.get("optimizer_status")
        vectors_count = result.get("vectors_count")
        indexed_vectors = result.get("indexed_vectors_count")
        points_count = result.get("points_count")
        segments_count = result.get("segments_count")
        print(f"  Collection status: {status}")
        print(f"  Optimizer status: {opt_status}")
        print(f"  Segments count: {segments_count}")
        print(f"  Total points: {points_count}")
        print(f"  Total vectors: {vectors_count}")
        print(f"  Indexed vectors: {indexed_vectors}")
    except Exception as e:
        print(f"  Failed to query Qdrant collection: {e}")

    # 5. SQLite Configuration & Schema
    print("\n[5. SQLite Configuration & Schema]")
    db_path = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
    print(f"  SQLite DB path: {db_path}")
    print(f"  SQLite DB exists: {db_path.exists()} ({db_path.stat().st_size / (1024*1024):.2f} MB)")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Check PRAGMAs
    pragmas = ["journal_mode", "synchronous", "cache_size", "mmap_size", "temp_store"]
    for p in pragmas:
        cur.execute(f"PRAGMA {p};")
        val = cur.fetchone()[0]
        print(f"  PRAGMA {p}: {val}")
    
    # Check indexes on passages
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='passages';")
    idx_rows = cur.fetchall()
    print(f"  Indexes on 'passages': {len(idx_rows)}")
    for name, sql in idx_rows:
        print(f"    - {name}: {sql}")
    conn.close()

    # 6. Filesystem Locations & OneDrive Check
    print("\n[6. Filesystem Locations & OneDrive Check]")
    data_dir = (REPO_ROOT / "data").resolve()
    print(f"  data/ directory: {data_dir}")
    is_data_onedrive = "onedrive" in str(data_dir).lower()
    print(f"  data/ inside OneDrive: {'YES (LATENCY IMPACT ALERT)' if is_data_onedrive else 'NO'}")

    print("=" * 60)

if __name__ == "__main__":
    inspect_profile()
