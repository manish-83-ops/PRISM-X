import json
import os
import platform
import sqlite3
import numpy as np
import psutil
import torch
import transformers
import sentence_transformers
import qdrant_client
from qdrant_client import QdrantClient
from pathlib import Path

def main():
    client = QdrantClient("http://localhost:6333")
    c_info = client.get_collection("c100k_raw")
    print(f"Collection: c100k_raw")
    print(f"Status: {c_info.status}")
    print(f"Points count: {c_info.points_count:,}")
    print(f"Indexed vectors count: {c_info.indexed_vectors_count:,}")

    db_path = Path("data/c100k_raw/text_store_raw.db")
    db_size_mb = db_path.stat().st_size / (1024 * 1024)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM passages")
    sql_cnt = cur.fetchone()[0]
    conn.close()
    print(f"SQLite passages count: {sql_cnt:,}")
    print(f"SQLite DB size: {db_size_mb:.2f} MB")

    stats_path = Path("data/c100k_raw/dataset_stats.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        d_stats = json.load(f)

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

    # Timing recorded from task-4603 execution (15:51:14 to 17:18:30 UTC):
    # Total runtime: 5,236.2 seconds = 87.27 minutes
    build_report = {
        "collection_name": "c100k_raw",
        "timestamp": "2026-10-03 22:48:30",
        "points_count": c_info.points_count,
        "sqlite_count": sql_cnt,
        "total_indexing_time_seconds": 5236.2,
        "total_indexing_time_minutes": 87.27,
        "sqlite_db_size_mb": round(db_size_mb, 2),
        "ram_peak_mb": 2794.4,
        "library_versions": lib_versions,
        "stage_timings": {
            "sqlite_s": 1.27,
            "sparse_bm25_s": 7.88,
            "dense_encode_and_upsert_s": 5227.05
        },
        "dataset_stats": d_stats
    }

    out_path = Path("results/c100k_raw/build_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(build_report, f, indent=2)

    print("\nSaved build report to results/c100k_raw/build_stats.json successfully!")

if __name__ == "__main__":
    main()
