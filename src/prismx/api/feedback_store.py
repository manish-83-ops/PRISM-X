"""Asynchronous Feedback Storage for PRISMX (Gate 15 - Tier B2).
Stores user search/passage relevance feedback (votes and queries) in an isolated
SQLite database (data/feedback.db). Never touches benchmark stores.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("prismx.feedback")

FEEDBACK_DB_PATH = Path("data/feedback.db")
FORBIDDEN_STORES = {"text_store_raw.db", "c100k_raw"}


def init_feedback_db(db_path: Path = FEEDBACK_DB_PATH) -> None:
    """Initialize feedback database and table if not existing."""
    if any(forbidden in str(db_path) for forbidden in FORBIDDEN_STORES):
        raise ValueError(f"CRITICAL INVARIANT VIOLATION: Cannot use benchmark store path {db_path} for feedback!")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id TEXT UNIQUE NOT NULL,
                query_id TEXT,
                query TEXT,
                passage_id TEXT NOT NULL,
                vote INTEGER NOT NULL,
                comment TEXT,
                client_ip TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_query ON user_feedback(query_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_passage ON user_feedback(passage_id);")
        conn.commit()


def record_feedback_async(
    feedback_id: str,
    query_id: str | None,
    query: str | None,
    passage_id: str,
    vote: int,
    comment: str | None = None,
    client_ip: str | None = None,
    db_path: Path = FEEDBACK_DB_PATH,
) -> None:
    """Worker task executed in FastAPI BackgroundTasks (non-blocking)."""
    if any(forbidden in str(db_path) for forbidden in FORBIDDEN_STORES):
        logger.error(f"Refusing to write feedback to forbidden path: {db_path}")
        return

    try:
        init_feedback_db(db_path)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            conn.execute(
                """
                INSERT INTO user_feedback (feedback_id, query_id, query, passage_id, vote, comment, client_ip, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (feedback_id, str(query_id) if query_id is not None else None, query, passage_id, vote, comment, client_ip, timestamp),
            )
            conn.commit()
        logger.info(f"Recorded feedback {feedback_id} for passage {passage_id} (vote={vote})")
    except Exception as exc:
        logger.error(f"Failed to write feedback asynchronously: {exc}")
