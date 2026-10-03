"""PRISMX Data Loader for MS MARCO Corpus and Splits."""

from __future__ import annotations

import os
from pathlib import Path
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

DEV_QUERIES_URL = "https://huggingface.co/datasets/Tevatron/msmarco-passage/resolve/main/dev.jsonl.gz"
DEV_QRELS_URL = "https://huggingface.co/datasets/BeIR/msmarco-qrels/resolve/main/dev.tsv"
CORPUS_URL = "https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus/resolve/main/corpus.jsonl.gz"

def ensure_raw_datasets() -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "dev_queries": RAW_DIR / "dev_queries.jsonl.gz",
        "dev_qrels": RAW_DIR / "dev_qrels.tsv",
        "corpus": RAW_DIR / "corpus.jsonl.gz",
    }
    
    if not paths["dev_queries"].exists():
        print(f"Downloading dev queries from {DEV_QUERIES_URL}...")
        urllib.request.urlretrieve(DEV_QUERIES_URL, paths["dev_queries"])
        
    if not paths["dev_qrels"].exists():
        print(f"Downloading dev qrels from {DEV_QRELS_URL}...")
        urllib.request.urlretrieve(DEV_QRELS_URL, paths["dev_qrels"])
        
    return paths

if __name__ == "__main__":
    p = ensure_raw_datasets()
    print("Raw datasets verified at:", p)
