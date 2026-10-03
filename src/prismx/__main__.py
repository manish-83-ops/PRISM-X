"""PRISMX CLI Entry Point."""

from __future__ import annotations

import argparse
import sys

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m prismx",
        description="PRISMX Vector Database and Hybrid RAG Engine CLI"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # search
    search_parser = subparsers.add_parser("search", help="Execute search query")
    search_parser.add_argument("--query", "-q", type=str, required=True, help="Search query string")
    search_parser.add_argument("--mode", "-m", choices=["dense", "hybrid"], default="dense", help="Retrieval mode")
    search_parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of passages to return")
    search_parser.add_argument("--category", "-c", type=str, default=None, help="Filter by category")

    # server
    server_parser = subparsers.add_parser("server", help="Launch PRISMX FastAPI HTTP server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    server_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Run ingestion pipeline")
    ingest_parser.add_argument("--config", type=str, default=None, help="Config YAML path")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    if args.subcommand == "search":
        print(f"Executing {args.mode} search for: '{args.query}' (top_k={args.top_k})")
    elif args.subcommand == "server":
        print(f"Launching PRISMX server on {args.host}:{args.port}...")
    elif args.subcommand == "ingest":
        print("Starting corpus ingestion pipeline...")

if __name__ == "__main__":
    main()
