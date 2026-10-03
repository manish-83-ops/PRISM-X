"""Gate 5: Serving Isolation and Zero-Leakage Verification Tests.

Verifies:
1. Serving modules (src/prismx/api/, src/prismx/retrieve/, src/prismx/index/) do not import eval modules.
2. Serving modules do not reference qrel files or split manifests.
3. Index-time pipelines (DenseEncoder, BM25Tokenizer, QdrantStore, TextStore) do not use query text.
4. Serving path runs without eval artifacts loaded into sys.modules.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "prismx"

SERVING_FILES = [
    SRC_DIR / "api" / "app.py",
    SRC_DIR / "schemas.py",
    SRC_DIR / "retrieve" / "service.py",
    SRC_DIR / "retrieve" / "dense.py",
    SRC_DIR / "retrieve" / "hybrid.py",
    SRC_DIR / "retrieve" / "fusion.py",
    SRC_DIR / "retrieve" / "rerank.py",
    SRC_DIR / "retrieve" / "cache.py",
    SRC_DIR / "index" / "encoder.py",
    SRC_DIR / "index" / "lexical.py",
    SRC_DIR / "index" / "qdrant_store.py",
    SRC_DIR / "index" / "text_store.py",
]


def test_no_eval_imports_in_serving_code():
    """Verify that no serving module imports from prismx.eval or eval."""
    violating_imports = []

    for py_file in SERVING_FILES:
        assert py_file.exists(), f"Serving file {py_file} does not exist"
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "eval" in alias.name:
                        violating_imports.append((py_file.name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "eval" in mod:
                    violating_imports.append((py_file.name, mod))

    assert not violating_imports, f"Serving modules must not import eval modules: {violating_imports}"


def test_no_qrel_or_split_references_in_serving_code():
    """Verify that serving code does not hardcode or access evaluation manifests/qrels."""
    forbidden_substrings = [
        "qrels",
        "split_bench",
        "split_tune",
        "split_test",
        "frozen_ragas",
        "gold_passage",
    ]

    violations = []
    for py_file in SERVING_FILES:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()

        for sub in forbidden_substrings:
            if sub in content:
                violations.append((py_file.name, sub))

    assert not violations, f"Forbidden benchmark/qrel references found in serving code: {violations}"


def test_index_time_pipeline_has_no_query_dependency():
    """Verify index-time encoders and tokenizers only operate on document/passage texts without query dependencies."""
    from prismx.index.encoder import DenseEncoder
    from prismx.index.lexical import BM25Tokenizer
    from prismx.index.qdrant_store import QdrantStore
    from prismx.index.text_store import TextStore

    # DenseEncoder passage indexing signature
    sig_enc = inspect.signature(DenseEncoder.encode_corpus_checkpointed)
    param_names = list(sig_enc.parameters.keys())
    assert "passages" in param_names
    assert "query" not in param_names
    assert "queries" not in param_names

    # BM25Tokenizer passage vector signature
    sig_tok = inspect.signature(BM25Tokenizer.compute_doc_sparse_vector)
    param_tok = list(sig_tok.parameters.keys())
    assert "text" in param_tok
    assert "query" not in param_tok

    # QdrantStore point indexing signature
    sig_qd = inspect.signature(QdrantStore.upsert_points_batch)
    param_qd = list(sig_qd.parameters.keys())
    assert "points" in param_qd
    assert "query" not in param_qd

    # TextStore indexing signature
    sig_ts = inspect.signature(TextStore.insert_passages_bulk)
    param_ts = list(sig_ts.parameters.keys())
    assert "passages" in param_ts
    assert "query" not in param_ts


def test_serving_runtime_does_not_load_eval_modules():
    """Verify that importing SearchService does not trigger importing prismx.eval."""
    # Ensure prismx.eval is not in sys.modules prior to check
    eval_mods = [k for k in sys.modules if "prismx.eval" in k]
    for k in eval_mods:
        del sys.modules[k]

    from prismx.retrieve.service import SearchService
    from prismx.schemas import SearchRequest

    assert "prismx.eval" not in sys.modules
    assert "prismx.eval.metrics" not in sys.modules
    assert "prismx.eval.bootstrap" not in sys.modules
