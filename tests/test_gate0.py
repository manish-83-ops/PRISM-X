"""Gate 0 Smoke and Environment Unit Tests."""

import sys
from pathlib import Path
import pytest
from qdrant_client import QdrantClient
from prismx.config import load_config, compute_config_hash

def test_python_version():
    assert sys.version_info >= (3, 10), "Python version must be >= 3.10"

def test_config_hash_determinism():
    cfg1 = {"b": 2, "a": 1, "c": [3, 2, 1]}
    cfg2 = {"a": 1, "c": [3, 2, 1], "b": 2}
    h1 = compute_config_hash(cfg1)
    h2 = compute_config_hash(cfg2)
    assert h1 == h2, "Canonical config hash must be independent of key order"
    assert len(h1) == 64, "SHA-256 hash must be 64 characters hex"

def test_config_loading():
    cfg = load_config()
    assert "_config_hash" in cfg
    assert cfg["encoder"]["embedding_dim"] == 384
    assert cfg["data"]["corpus_size"] == 100000

@pytest.mark.integration
def test_qdrant_connectivity():
    client = QdrantClient(url="http://127.0.0.1:6333", grpc_port=6334, prefer_grpc=True)
    cols = client.get_collections()
    assert hasattr(cols, "collections")
