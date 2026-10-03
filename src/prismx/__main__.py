"""PRISMX Unified CLI Entry Point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prismx.cli")


def cmd_search(args: argparse.Namespace) -> None:
    """Execute search query via SearchService."""
    from prismx.config import load_config
    from prismx.index.encoder import DenseEncoder
    from prismx.index.lexical import BM25Tokenizer
    from prismx.index.qdrant_store import QdrantStore
    from prismx.index.text_store import TextStore
    from prismx.retrieve.dense import DenseRetriever
    from prismx.retrieve.hybrid import HybridRetriever
    from prismx.retrieve.service import SearchService
    from prismx.schemas import FilterParams, FusionParams, SearchRequest

    cfg = load_config(args.config)
    text_store = TextStore(db_path=cfg["sqlite"]["db_path"])
    qdrant_store = QdrantStore(
        host=cfg["qdrant"]["host"],
        port=cfg["qdrant"]["port"],
        grpc_port=cfg["qdrant"]["grpc_port"],
        prefer_grpc=cfg["qdrant"].get("prefer_grpc", True),
        collection_name=cfg["qdrant"]["collection_name"],
    )
    encoder = DenseEncoder(
        model_name=cfg["encoder"]["model_name"],
        embedding_dim=cfg["encoder"]["embedding_dim"],
        max_seq_length=cfg["encoder"]["max_seq_length"],
        torch_threads=cfg["encoder"].get("torch_threads", 12),
    )
    tokenizer = BM25Tokenizer(
        k1=cfg["lexical"]["k1"],
        b=cfg["lexical"]["b"],
        use_stemming=cfg["lexical"].get("stemming", False),
    )

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

    filters = None
    if args.category or args.source:
        filters = FilterParams(category=args.category, source=args.source)

    fusion = None
    if args.fusion_method:
        fusion = FusionParams(
            method=args.fusion_method,
            alpha=args.alpha if args.alpha is not None else 0.7,
            rrf_k=args.rrf_k if args.rrf_k is not None else 60,
        )

    req = SearchRequest(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        filters=filters,
        fusion=fusion,
    )

    resp = service.search(req)

    print("\n" + "=" * 78)
    print(f"PRISMX SEARCH RESULTS: mode={resp.mode} | query='{resp.query}'")
    print(f"Index Version: {resp.index_version} | Latencies: total={resp.latency_ms.total}ms (enc={resp.latency_ms.encode}ms, dense={resp.latency_ms.dense}ms, sparse={resp.latency_ms.sparse}ms, fusion={resp.latency_ms.fusion}ms, fetch={resp.latency_ms.fetch_text}ms)")
    if resp.filters_applied:
        print(f"Filters Applied: {resp.filters_applied}")
    print("=" * 78)

    for item in resp.results:
        snippet = item.text[:120].replace("\n", " ") + ("..." if len(item.text) > 120 else "")
        print(f"\n[Rank {item.rank}] Passage ID: {item.passage_id} | Score: {item.score:.4f} | Cat: {item.category}")
        if resp.mode == "hybrid":
            dense_info = f"rank={item.dense_rank}, score={item.dense_score:.4f}" if item.dense_rank else "none"
            bm25_info = f"rank={item.bm25_rank}, score={item.bm25_score:.4f}" if item.bm25_rank else "none"
            print(f"         Dense: {dense_info} | BM25: {bm25_info}")
        print(f"         Text: {snippet}")

    print("\n" + "=" * 78)
    text_store.close()
    qdrant_store.close()


def cmd_server(args: argparse.Namespace) -> None:
    """Launch FastAPI HTTP server."""
    import uvicorn
    print(f"Starting PRISMX FastAPI server on {args.host}:{args.port}...")
    uvicorn.run("prismx.api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run corpus ingestion pipeline."""
    from prismx.index.ingest import run_ingest
    run_ingest(config_path=args.config)


def cmd_reindex_sparse(args: argparse.Namespace) -> None:
    """Reindex sparse vectors with updated avgdl_ref (PATCH-1)."""
    from prismx.index.reindex import reindex_sparse_vectors
    reindex_sparse_vectors(threshold=args.threshold, force=args.force, batch_size=args.batch_size)


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation benchmark."""
    from prismx.eval.runner import run_eval_pipeline
    run_eval_pipeline(split=args.split, mode=args.mode, config_path=args.config)


def cmd_bench(args: argparse.Namespace) -> None:
    """Run latency benchmarks across threads."""
    from prismx.eval.bench import run_latency_benchmark
    threads = [int(t.strip()) for t in args.threads.split(",")]
    run_latency_benchmark(threads_list=threads, mode=args.mode, config_path=args.config)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m prismx",
        description="PRISMX Vector Database and Hybrid RAG Engine CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # search
    search_p = subparsers.add_parser("search", help="Execute search query")
    search_p.add_argument("--query", "-q", type=str, required=True, help="Search query string")
    search_p.add_argument("--mode", "-m", choices=["dense", "hybrid"], default="dense", help="Retrieval mode")
    search_p.add_argument("--top_k", "-k", type=int, default=5, help="Number of passages to return")
    search_p.add_argument("--category", "-c", type=str, default=None, help="Filter by category")
    search_p.add_argument("--source", "-s", type=str, default=None, help="Filter by source")
    search_p.add_argument("--fusion_method", choices=["weighted", "rrf"], default=None, help="Fusion method")
    search_p.add_argument("--alpha", type=float, default=None, help="Dense weight alpha (for weighted fusion)")
    search_p.add_argument("--rrf_k", type=int, default=None, help="RRF smoothing constant k")
    search_p.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    # server
    server_p = subparsers.add_parser("server", help="Launch PRISMX FastAPI HTTP server")
    server_p.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    server_p.add_argument("--port", type=int, default=8000, help="Port number")
    server_p.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Run ingestion pipeline")
    ingest_p.add_argument("--config", type=str, default=None, help="Config YAML path")

    # reindex-sparse
    reindex_p = subparsers.add_parser("reindex-sparse", help="Reindex sparse BM25 vectors (PATCH-1)")
    reindex_p.add_argument("--threshold", type=float, default=0.10, help="Drift threshold fraction (default: 0.10)")
    reindex_p.add_argument("--force", action="store_true", help="Force reindexing even if drift <= threshold")
    reindex_p.add_argument("--batch_size", type=int, default=1000, help="Batch size for vector updates")

    # eval
    eval_p = subparsers.add_parser("eval", help="Run evaluation pipeline")
    eval_p.add_argument("--split", choices=["tune", "test", "ragas"], default="tune", help="Evaluation split")
    eval_p.add_argument("--mode", choices=["dense", "hybrid"], default="dense", help="Retrieval mode")
    eval_p.add_argument("--config", type=str, default=None, help="Config YAML path")

    # bench
    bench_p = subparsers.add_parser("bench", help="Run latency benchmark across threads")
    bench_p.add_argument("--mode", choices=["dense", "hybrid"], default="dense", help="Retrieval mode")
    bench_p.add_argument("--threads", type=str, default="1,4", help="Comma-separated thread counts, e.g. '1,4'")
    bench_p.add_argument("--config", type=str, default=None, help="Config YAML path")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "search": cmd_search,
        "server": cmd_server,
        "ingest": cmd_ingest,
        "reindex-sparse": cmd_reindex_sparse,
        "eval": cmd_eval,
        "bench": cmd_bench,
    }

    fn = dispatch.get(args.subcommand)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
