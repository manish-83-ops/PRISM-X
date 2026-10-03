"""PRISMX SQLite Decoupled Text Store with O(1) Length Tracking and Versioning."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "text_store.db"

class TextStore:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def close(self) -> None:
        """No-op for connection-per-call SQLite store, provided for lifecycle compatibility."""
        pass

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS passages (
                    passage_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    category TEXT,
                    source TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        val_str = json.dumps(value) if not isinstance(value, str) else value
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (key, val_str)
            )
            conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT value FROM meta WHERE key = ?;", (key,))
            row = cur.fetchone()
            if row is None:
                return default
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]

    def insert_passages_bulk(self, passages: list[dict[str, Any]], batch_size: int = 5000) -> None:
        """Bulk inserts passages into SQLite."""
        with self._get_connection() as conn:
            for i in range(0, len(passages), batch_size):
                batch = passages[i : i + batch_size]
                conn.executemany(
                    "INSERT OR REPLACE INTO passages (passage_id, text, category, source) VALUES (?, ?, ?, ?);",
                    [
                        (
                            str(p["passage_id"]),
                            p["text"],
                            p.get("category", "general"),
                            p.get("source", "msmarco"),
                        )
                        for p in batch
                    ],
                )
            conn.commit()

    def get_passages_by_ids(self, passage_ids: list[str], chunk_size: int = 400) -> dict[str, dict[str, Any]]:
        """Fetches passages by ID in chunks <= 500, returning a dict keyed by passage_id.
        
        Adheres to PATCH-2: SQL order is never relied upon; results returned as a dict.
        """
        if not passage_ids:
            return {}

        results: dict[str, dict[str, Any]] = {}
        with self._get_connection() as conn:
            for i in range(0, len(passage_ids), chunk_size):
                chunk = passage_ids[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                query = f"SELECT passage_id, text, category, source FROM passages WHERE passage_id IN ({placeholders});"
                cur = conn.execute(query, chunk)
                for row in cur.fetchall():
                    pid, text, cat, src = row
                    results[str(pid)] = {
                        "passage_id": str(pid),
                        "text": text,
                        "category": cat,
                        "source": src,
                    }
        return results

    def get_passage_count(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM passages;")
            return int(cur.fetchone()[0])

    def upsert_single(
        self,
        passage_id: str,
        text: str,
        category: str | None = None,
        source: str | None = None,
        doc_token_len: int = 0,
    ) -> int:
        """Upserts a single passage with O(1) total_doc_len tracking and index_version increment.
        
        Returns the new index_version.
        """
        passage_id = str(passage_id)
        with self._get_connection() as conn:
            # Check if passage already exists to calculate length delta
            cur = conn.execute("SELECT text FROM passages WHERE passage_id = ?;", (passage_id,))
            existing = cur.fetchone()

            conn.execute(
                "INSERT INTO passages (passage_id, text, category, source) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(passage_id) DO UPDATE SET text=excluded.text, category=excluded.category, source=excluded.source;",
                (passage_id, text, category or "general", source or "manual"),
            )

            # Update n_docs and total_doc_len in meta
            n_docs = int(self.get_meta("n_docs", 0))
            total_doc_len = int(self.get_meta("total_doc_len", 0))
            index_version = int(self.get_meta("index_version", 1))

            if existing is None:
                n_docs += 1
                total_doc_len += doc_token_len
            else:
                # Delta length approximation
                prev_len = len(existing[0].split())
                total_doc_len = max(0, total_doc_len - prev_len + doc_token_len)

            index_version += 1

            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('n_docs', ?);", (str(n_docs),))
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('total_doc_len', ?);", (str(total_doc_len),))
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('index_version', ?);", (str(index_version),))
            conn.commit()

        return index_version

    def delete_single(self, passage_id: str) -> tuple[bool, int]:
        """Deletes a passage with O(1) length tracking and index_version increment.
        
        Returns (was_deleted, new_index_version).
        """
        passage_id = str(passage_id)
        with self._get_connection() as conn:
            cur = conn.execute("SELECT text FROM passages WHERE passage_id = ?;", (passage_id,))
            row = cur.fetchone()
            if row is None:
                return False, int(self.get_meta("index_version", 1))

            doc_len = len(row[0].split())
            conn.execute("DELETE FROM passages WHERE passage_id = ?;", (passage_id,))

            n_docs = max(0, int(self.get_meta("n_docs", 1)) - 1)
            total_doc_len = max(0, int(self.get_meta("total_doc_len", 0)) - doc_len)
            index_version = int(self.get_meta("index_version", 1)) + 1

            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('n_docs', ?);", (str(n_docs),))
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('total_doc_len', ?);", (str(total_doc_len),))
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('index_version', ?);", (str(index_version),))
            conn.commit()

        return True, index_version

    def get_stats(self) -> dict[str, Any]:
        """Calculates true_avgdl and drift = |true_avgdl - avgdl_ref| / avgdl_ref."""
        avgdl_ref = float(self.get_meta("avgdl_ref", 50.0))
        n_docs = int(self.get_meta("n_docs", 0))
        total_doc_len = int(self.get_meta("total_doc_len", 0))
        index_version = int(self.get_meta("index_version", 1))

        true_avgdl = (total_doc_len / n_docs) if n_docs > 0 else avgdl_ref
        drift = abs(true_avgdl - avgdl_ref) / (avgdl_ref if avgdl_ref > 0 else 1.0)

        return {
            "n_docs": n_docs,
            "total_doc_len": total_doc_len,
            "avgdl_ref": round(avgdl_ref, 4),
            "true_avgdl": round(true_avgdl, 4),
            "drift": round(drift, 4),
            "index_version": index_version,
            "drift_exceeds_threshold": drift > 0.10,
        }
