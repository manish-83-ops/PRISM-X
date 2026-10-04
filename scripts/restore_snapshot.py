#!/usr/bin/env python3
"""
PRISM-X Qdrant Snapshot Recovery Utility
Restores the frozen c100k_raw index (100,008 points) into Qdrant from snapshot.
Compatible with local native Qdrant and Dockerized Qdrant.
"""

import argparse
import glob
import os
import sys
import time
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


def main():
    parser = argparse.ArgumentParser(description="Restore PRISM-X Qdrant snapshot")
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Path to .snapshot file. Defaults to latest snapshot in snapshots/c100k_raw/",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="c100k_raw",
        help="Target collection name (default: c100k_raw)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:6333",
        help="Qdrant HTTP endpoint (default: http://127.0.0.1:6333)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restore even if collection already exists",
    )
    args = parser.parse_args()

    snapshot_path = args.snapshot
    if not snapshot_path:
        candidates = sorted(glob.glob("snapshots/c100k_raw/*.snapshot"))
        if not candidates:
            # Check data/qdrant_snapshots/
            candidates = sorted(glob.glob("data/qdrant_snapshots/*.snapshot"))
        if candidates:
            snapshot_path = candidates[-1]

    if not snapshot_path or not os.path.exists(snapshot_path):
        print(f"[ERROR] Snapshot file not found: {snapshot_path}")
        print("Please place the c100k_raw snapshot in snapshots/c100k_raw/ or specify --snapshot <path>")
        sys.exit(1)

    print(f"[*] Snapshot source: {snapshot_path} ({os.path.getsize(snapshot_path) / (1024*1024):.1f} MB)")
    print(f"[*] Target collection: {args.collection}")
    print(f"[*] Qdrant endpoint: {args.url}")

    client = QdrantClient(args.url, timeout=300.0)
    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception as e:
        print(f"[ERROR] Could not connect to Qdrant at {args.url}: {e}")
        print("Ensure Qdrant is running via 'docker compose up -d' or './bin/qdrant.exe --config-path config/qdrant.yaml'")
        sys.exit(1)

    if args.collection in collections:
        info = client.get_collection(args.collection)
        print(f"[i] Collection '{args.collection}' already exists with {info.points_count} points (status: {info.status}).")
        if not args.force:
            print("[i] Skipping restore since collection exists. Use --force to overwrite.")
            sys.exit(0)
        print("[!] Overwriting existing collection with snapshot...")

    abs_path = os.path.abspath(snapshot_path).replace("\\", "/")
    file_url = f"file:///{abs_path}"
    print(f"[*] Restoring snapshot from: {file_url}")
    t0 = time.time()
    try:
        client.recover_snapshot(
            collection_name=args.collection,
            location=file_url,
            priority="snapshot",
            wait=True,
        )
    except Exception as e:
        print(f"[ERROR] Snapshot recovery failed: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    info = client.get_collection(args.collection)
    print(f"[SUCCESS] Collection '{args.collection}' restored in {elapsed:.2f}s!")
    print(f"          Points: {info.points_count} | Status: {info.status}")
    if info.points_count == 100008:
        print("[PASS] Point count verified: exactly 100,008 passages.")
    else:
        print(f"[WARN] Expected 100,008 points, got {info.points_count}.")


if __name__ == "__main__":
    main()
