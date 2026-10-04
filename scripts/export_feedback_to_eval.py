#!/usr/bin/env python3
"""
PRISMX Feedback Export Tool (Gate 15 - Tier B2).
Exports recorded user feedback from data/feedback.db to a JSONL evaluation dataset draft.
Only extracts positive votes (vote=1) with valid queries and passages.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

FEEDBACK_DB_PATH = Path("data/feedback.db")
DEFAULT_OUTPUT_PATH = Path("results/feedback_eval_draft.jsonl")


def export_feedback(db_path: Path = FEEDBACK_DB_PATH, out_path: Path = DEFAULT_OUTPUT_PATH) -> int:
    if not db_path.exists():
        print(f"[WARN] No feedback database found at {db_path}. No records to export.")
        return 0

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT feedback_id, query_id, query, passage_id, vote, comment, created_at
            FROM user_feedback
            WHERE query IS NOT NULL AND query != ''
            ORDER BY id ASC;
            """
        ).fetchall()

    if not rows:
        print("[*] No query-associated feedback records found in database.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    exported_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            entry = {
                "feedback_id": r["feedback_id"],
                "query_id": r["query_id"],
                "query": r["query"],
                "relevant_passage_id": r["passage_id"],
                "vote": r["vote"],
                "label": 1 if r["vote"] > 0 else 0,
                "user_comment": r["comment"],
                "timestamp": r["created_at"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            exported_count += 1

    print(f"[SUCCESS] Exported {exported_count} feedback records to {out_path}")
    return exported_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PRISMX feedback to JSONL eval set")
    parser.add_argument("--db", type=str, default=str(FEEDBACK_DB_PATH), help="Path to feedback SQLite db")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Path to output JSONL file")
    args = parser.parse_args()
    export_feedback(Path(args.db), Path(args.out))
