"""PRISMX FastAPI Backend Application."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from prismx.config import load_config
from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import QdrantStore
from prismx.index.text_store import TextStore
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.service import SearchService
from prismx.schemas import (
    DeleteResponse,
    ErrorResponse,
    MetaResponse,
    SearchRequest,
    SearchResponse,
    UpsertRequest,
    UpsertResponse,
)

logger = logging.getLogger("prismx.api")

# Global state container
_state: dict[str, Any] = {
    "config": None,
    "text_store": None,
    "qdrant_store": None,
    "encoder": None,
    "tokenizer": None,
    "service": None,
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown initialization."""
    logger.info("Initializing PRISMX backend...")
    cfg = load_config()
    _state["config"] = cfg

    # 1. TextStore (SQLite)
    text_store = TextStore(db_path=cfg["sqlite"]["db_path"])
    _state["text_store"] = text_store

    # 2. QdrantStore
    qdrant_store = QdrantStore(
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        grpc_port=cfg["qdrant"]["grpc_port"],
        prefer_grpc=cfg["qdrant"].get("prefer_grpc", True),
        collection_name=cfg["qdrant"]["collection_name"],
    )
    _state["qdrant_store"] = qdrant_store

    # 3. Encoder & Tokenizer
    encoder = DenseEncoder(
        model_name=cfg["encoder"]["model_name"],
        embedding_dim=cfg["encoder"]["embedding_dim"],
        max_seq_length=cfg["encoder"]["max_seq_length"],
        torch_threads=cfg["encoder"].get("torch_threads", 12),
    )
    _state["encoder"] = encoder

    tokenizer = BM25Tokenizer(
        k1=cfg["lexical"]["k1"],
        b=cfg["lexical"]["b"],
        use_stemming=cfg["lexical"].get("stemming", False),
    )
    _state["tokenizer"] = tokenizer

    # 4. Retrievers & SearchService
    dense_retriever = DenseRetriever(encoder=encoder, qdrant_store=qdrant_store, config=cfg)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        tokenizer=tokenizer,
        qdrant_store=qdrant_store,
        config=cfg,
    )
    service = SearchService(
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        text_store=text_store,
        qdrant_store=qdrant_store,
        config=cfg,
    )
    _state["service"] = service

    # 5. Warm up pipeline
    try:
        logger.info("Warming up models with test query...")
        encoder.encode_queries(["warmup query"])
        _state["ready"] = True
        logger.info("PRISMX backend initialization complete and ready.")
    except Exception as exc:
        logger.warning(f"Warmup encountered an issue: {exc}")
        _state["ready"] = True

    yield

    logger.info("Shutting down PRISMX backend...")
    if _state["text_store"]:
        _state["text_store"].close()
    if _state["qdrant_store"]:
        _state["qdrant_store"].close()
    logger.info("PRISMX backend shutdown complete.")


app = FastAPI(
    title="PRISMX Vector Database and Hybrid RAG Engine",
    description="High-performance dual-vector retrieval system with BM25 sparse IDF and dense embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for teammate UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "VALIDATION_ERROR", "detail": str(exc.errors())},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP_ERROR", "detail": exc.detail},
    )


def get_service() -> SearchService:
    service = _state.get("service")
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PRISMX service is not initialized yet.",
        )
    return service


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/ready", tags=["System"])
async def readiness_check() -> dict[str, Any]:
    """Readiness probe."""
    is_ready = bool(_state.get("ready", False))
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is warming up.",
        )
    return {
        "status": "ready",
        "models_loaded": _state.get("encoder") is not None,
        "qdrant_connected": _state.get("qdrant_store") is not None,
        "warmed_up": True,
    }


@app.get("/meta", response_model=MetaResponse, tags=["System"])
async def get_meta() -> MetaResponse:
    """Returns system metadata, index statistics, drift metrics, and categories."""
    service = get_service()
    meta = service.get_meta()
    categories = []
    cluster_meta_file = Path("data/manifests/cluster_metadata.json")
    if cluster_meta_file.exists():
        try:
            with open(cluster_meta_file, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                categories = list(c_data.get("cluster_labels", {}).values())
        except Exception:
            pass
    if not categories:
        categories = ["calories-food", "click-use", "tax-state", "cost-average", "symptoms-pain"]

    return MetaResponse(
        modes=meta["modes"],
        fusion_defaults=meta["fusion_defaults"],
        point_count=meta["point_count"],
        sqlite_count=meta.get("sqlite_count"),
        index_version=meta["index_version"],
        avgdl_ref=meta.get("avgdl_ref"),
        true_avgdl=meta.get("true_avgdl"),
        drift=meta.get("drift"),
        drift_warning=meta.get("drift_warning"),
        inconsistency_count=meta.get("inconsistency_count", 0),
        categories=categories,
        sources=["msmarco-passage", "manual"],
        models=meta["models"],
        config_hash=meta["config_hash"],
    )


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(req: SearchRequest) -> SearchResponse:
    """Execute dense or hybrid retrieval with optional metadata pre-filtering."""
    service = get_service()
    try:
        return service.search(req)
    except Exception as exc:
        logger.exception("Error executing search request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        )


@app.post("/passages/upsert", response_model=UpsertResponse, tags=["Ingestion"])
async def upsert_passage(req: UpsertRequest) -> UpsertResponse:
    """Atomic upsert of passage into both Qdrant and SQLite with index version bump."""
    service = get_service()
    try:
        return service.upsert_passage(req)
    except Exception as exc:
        logger.exception("Error upserting passage")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upsert failed: {exc}",
        )


@app.delete("/passages/{passage_id}", response_model=DeleteResponse, tags=["Ingestion"])
async def delete_passage(passage_id: str) -> DeleteResponse:
    """Delete passage from Qdrant and SQLite."""
    service = get_service()
    try:
        return service.delete_passage(passage_id)
    except Exception as exc:
        logger.exception("Error deleting passage")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {exc}",
        )


@app.get("/bench/latest", tags=["Benchmarks"])
async def get_latest_benchmark() -> dict[str, Any]:
    """Retrieve the latest latency benchmark results."""
    bench_file = Path("results/phase2/latency_benchmark.json")
    if not bench_file.exists():
        # Fallback to phase1 if phase2 not run yet
        bench_file = Path("results/phase1/latency_benchmark.json")

    if not bench_file.exists():
        return {"status": "no_benchmark_run_yet", "results": None}

    with open(bench_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/eval/latest", tags=["Evaluation"])
async def get_latest_evaluation() -> dict[str, Any]:
    """Retrieve the latest retrieval evaluation metrics."""
    eval_file = Path("results/phase2/eval_results.json")
    if not eval_file.exists():
        eval_file = Path("results/phase1/eval_results.json")

    if not eval_file.exists():
        return {"status": "no_eval_run_yet", "results": None}

    with open(eval_file, "r", encoding="utf-8") as f:
        return json.load(f)
