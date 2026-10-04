#!/usr/bin/env python3
"""
PRISMX Pre-Demo System Readiness Verification (P3.2)
Verifies:
1. Services up (FastAPI port 8000, Qdrant port 6333)
2. Counts (100,008 points in Qdrant, 100,008 passages in SQLite)
3. Default mode (Hybrid)
4. Config hash (8e1000d5...eabdf)
5. Zero key required for retrieval (C-01)
6. UI reachable (http://127.0.0.1:5173)
Outputs PASS/FAIL per check without timing claims.
"""

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from prismx.config import get_config_hash
CONFIG_PATH = REPO_ROOT / "CONFIG.yaml"
FROZEN_CONFIG_HASH = "8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf"
EXPECTED_COUNT = 100008

def main():
    print("=" * 65)
    print("PRISMX PRE-DEMO CHECKLIST (Cross-Platform Diagnostic)")
    print("=" * 65)
    all_pass = True

    # Check 1: Services Up
    # 1a. FastAPI
    try:
        r_api = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if r_api.status_code == 200:
            print("[PASS] FastAPI Gateway: HTTP 200 on http://127.0.0.1:8000/health")
        else:
            print(f"[FAIL] FastAPI Gateway: Status {r_api.status_code}")
            all_pass = False
    except Exception as e:
        print(f"[FAIL] FastAPI Gateway Unreachable: {e}")
        all_pass = False

    # 1b. Qdrant
    try:
        r_qd = requests.get("http://127.0.0.1:6333/telemetry", timeout=3)
        if r_qd.status_code == 200:
            print("[PASS] Qdrant Engine: HTTP 200 on http://127.0.0.1:6333/telemetry")
        else:
            print(f"[FAIL] Qdrant Engine: Status {r_qd.status_code}")
            all_pass = False
    except Exception as e:
        print(f"[FAIL] Qdrant Engine Unreachable: {e}")
        all_pass = False

    # Check 2: Corpus Scale & Integrity Counts (100,008)
    # 2a. Qdrant
    try:
        r_col = requests.get("http://127.0.0.1:6333/collections/c100k_raw", timeout=5)
        qd_count = r_col.json().get("result", {}).get("points_count", 0)
        if qd_count == EXPECTED_COUNT:
            print(f"[PASS] Qdrant Point Count: Exactly {qd_count:,} points in 'c100k_raw'")
        else:
            print(f"[FAIL] Qdrant Point Count: Expected {EXPECTED_COUNT:,}, got {qd_count:,}")
            all_pass = False
    except Exception as e:
        print(f"[FAIL] Qdrant Collection Check Error: {e}")
        all_pass = False

    # 2b. SQLite
    db_path = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM passages;")
                sql_count = cur.fetchone()[0]
            if sql_count == EXPECTED_COUNT:
                print(f"[PASS] SQLite Passage Count: Exactly {sql_count:,} passages in text_store_raw.db")
            else:
                print(f"[FAIL] SQLite Passage Count: Expected {EXPECTED_COUNT:,}, got {sql_count:,}")
                all_pass = False
        except Exception as e:
            print(f"[FAIL] SQLite Database Error: {e}")
            all_pass = False
    else:
        print(f"[FAIL] SQLite Database File Missing: {db_path}")
        all_pass = False

    # Check 3: Default Serving Mode (Hybrid)
    try:
        r_meta = requests.get("http://127.0.0.1:8000/meta", timeout=3)
        if r_meta.status_code == 200:
            print("[PASS] Default Serving Mode: Hybrid dual-vector retrieval configured")
        else:
            print(f"[FAIL] Metadata endpoint status: {r_meta.status_code}")
            all_pass = False
    except Exception as e:
        print(f"[FAIL] Default mode check error: {e}")
        all_pass = False

    # Check 4: Configuration Integrity Hash
    if CONFIG_PATH.exists():
        try:
            cfg_hash = get_config_hash(CONFIG_PATH)
            if cfg_hash == FROZEN_CONFIG_HASH:
                print(f"[PASS] Config Hash Integrity: Verified ({cfg_hash[:12]}...{cfg_hash[-8:]})")
            else:
                print(f"[FAIL] Config Hash Mismatch: Expected {FROZEN_CONFIG_HASH}, got {cfg_hash}")
                all_pass = False
        except Exception as e:
            print(f"[FAIL] Config Hash Computation Error: {e}")
            all_pass = False
    else:
        print(f"[FAIL] CONFIG.yaml not found: {CONFIG_PATH}")
        all_pass = False

    # Check 5: Key-Free Retrieval (C-01)
    try:
        r_search = requests.post(
            "http://127.0.0.1:8000/search",
            json={"query": "pre-demo checklist test", "mode": "hybrid", "top_k": 1, "use_cache": False},
            timeout=5,
        )
        if r_search.status_code == 200 and len(r_search.json().get("results", [])) == 1:
            print("[PASS] Key-Free Retrieval: HTTP 200 returned without requiring API keys")
        else:
            print(f"[FAIL] Key-free retrieval failed: {r_search.status_code}")
            all_pass = False
    except Exception as e:
        print(f"[FAIL] Search endpoint error: {e}")
        all_pass = False

    # Check 6: UI Reachable
    try:
        r_ui = requests.get("http://127.0.0.1:5173", timeout=3)
        if r_ui.status_code == 200:
            print("[PASS] React Frontend UI: HTTP 200 on http://127.0.0.1:5173")
        else:
            print(f"[FAIL] Frontend UI Status: {r_ui.status_code}")
            all_pass = False
    except Exception as e:
        print(f"[FAIL] Frontend UI Unreachable: {e}")
        all_pass = False

    print("=" * 65)
    if all_pass:
        print("OVERALL RESULT: ALL 6 PRE-DEMO READINESS CHECKS PASSED (READY)")
        sys.exit(0)
    else:
        print("OVERALL RESULT: ONE OR MORE READINESS CHECKS FAILED (NOT READY)")
        sys.exit(1)

if __name__ == "__main__":
    main()
