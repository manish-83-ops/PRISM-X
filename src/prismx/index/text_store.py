"""PRISMX SQLite Decoupled Text Store with Dual-Write Outbox and O(1) Versioning.
Conforms to ADR-024:
- SQLite is the authoritative source of truth.
- Dual-Write Outbox Pattern: Every mutation (upsert/delete) atomically writes the row
  and an outbox operation ('pending') in the same SQLite transaction.
- Idempotent application to Qdrant using deterministic point IDs.
- Mark applied upon successful Qdrant write.
- Startup replay: Unapplied pending operations are replayed on service initialization.
- Consistency probe: /meta exposes pending count and consistency probe across counts and sampled IDs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import random
import sqlite3
from typing import Any

logger = logging.getLogger("prismx.text_store")

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
        conn.execute("PRAGMA mmap_size=268435456;")
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outbox_ops (
                    op_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    passage_id TEXT NOT NULL,
                    op_type TEXT NOT NULL,
                    payload TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    applied_at TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox_ops(status);")
            conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        val_str = json.dumps(value) if not isinstance(value, str) else value
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (key, val_str),
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
        """Fetches passages by ID in chunks <= 500, returning a dict keyed by passage_id."""
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
        """Upserts a single passage with atomic outbox write and index_version increment.
        
        Returns the new index_version.
        """
        op_id, index_version = self.upsert_with_outbox(
            passage_id=passage_id,
            text=text,
            category=category,
            source=source,
            doc_token_len=doc_token_len,
        )
        return index_version

    def upsert_with_outbox(
        self,
        passage_id: str,
        text: str,
        category: str | None = None,
        source: str | None = None,
        doc_token_len: int = 0,
    ) -> tuple[int, int]:
        """Atomically writes passage and pending outbox operation in a single SQLite transaction.
        
        Returns (op_id, new_index_version).
        """
        passage_id = str(passage_id)
        cat = category or "general"
        src = source or "manual"
        payload_json = json.dumps({"text": text, "category": cat, "source": src})

        with self._get_connection() as conn:
            cur = conn.execute("SELECT text FROM passages WHERE passage_id = ?;", (passage_id,))
            existing = cur.fetchone()

            conn.execute(
                "INSERT INTO passages (passage_id, text, category, source) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(passage_id) DO UPDATE SET text=excluded.text, category=excluded.category, source=excluded.source;",
                (passage_id, text, cat, src),
            )

            # Insert atomic outbox operation
            cur_op = conn.execute(
                "INSERT INTO outbox_ops (passage_id, op_type, payload, status) VALUES (?, 'upsert', ?, 'pending');",
                (passage_id, payload_json),
            )
            op_id = cur_op.lastrowid

            n_docs = int(self.get_meta("n_docs", 0))
            total_doc_len = int(self.get_meta("total_doc_len", 0))
            index_version = int(self.get_meta("index_version", 1))

            if existing is None:
                n_docs += 1
                total_doc_len += doc_token_len
            else:
                prev_len = len(existing[0].split())
                total_doc_len = max(0, total_doc_len - prev_len + doc_token_len)

            index_version += 1

            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('n_docs', ?);", (str(n_docs),))
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('total_doc_len', ?);", (str(total_doc_len),))
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('index_version', ?);", (str(index_version),))
            conn.commit()

        return op_id, index_version

    def delete_single(self, passage_id: str) -> tuple[bool, int]:
        """Deletes a passage with atomic outbox write and index_version increment.
        
        Returns (was_deleted, new_index_version).
        """
        op_id, was_deleted, index_version = self.delete_with_outbox(passage_id)
        return was_deleted, index_version

    def delete_with_outbox(self, passage_id: str) -> tuple[int, bool, int]:
        """Atomically deletes passage and writes pending outbox operation in a single SQLite transaction.
        
        Returns (op_id, was_deleted, new_index_version).
        """
        passage_id = str(passage_id)
        with self._get_connection() as conn:
            cur = conn.execute("SELECT text FROM passages WHERE passage_id = ?;", (passage_id,))
            row = cur.fetchone()
            if row is None:
                return -1, False, int(self.get_meta("index_version", 1))

            doc_len = len(row[0].split())
            conn.execute("DELETE FROM passages WHERE passage_id = ?;", (passage_id,))

            cur_op = conn.execute(
                "INSERT INTO outbox_ops (passage_id, op_type, payload, status) VALUES (?, 'delete', NULL, 'pending');",
                (passage_id,),
            )
            op_id = cur_op.lastrowid

            n_docs = max(0, int(self.get_meta("n_docs", 1)) - 1)
            total_doc_len = max(0, int(self.get_meta("total_doc_len", 0)) - doc_len)
            index_version = int(self.get_meta("index_version", 1)) + 1

            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('n_docs', ?);", (str(n_docs),))
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('total_doc_len', ?);", (str(total_doc_len),))
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('index_version', ?);", (str(index_version),))
            conn.commit()

        return op_id, True, index_version

    def mark_outbox_applied(self, op_id: int) -> None:
        """Marks an outbox operation as successfully applied to downstream stores."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE outbox_ops SET status = 'applied', applied_at = CURRENT_TIMESTAMP WHERE op_id = ?;",
                (op_id,),
            )
            conn.commit()

    def get_pending_outbox_ops(self) -> list[dict[str, Any]]:
        """Returns all unapplied pending outbox operations in FIFO order."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT op_id, passage_id, op_type, payload, created_at FROM outbox_ops WHERE status = 'pending' ORDER BY op_id ASC;"
            )
            return [
                {
                    "op_id": r[0],
                    "passage_id": r[1],
                    "op_type": r[2],
                    "payload": json.loads(r[3]) if r[3] else None,
                    "created_at": r[4],
                }
                for r in cur.fetchall()
            ]

    def get_outbox_pending_count(self) -> int:
        """Returns the number of pending outbox operations."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM outbox_ops WHERE status = 'pending';")
            return int(cur.fetchone()[0])

    def consistency_probe(self, qdrant_store: Any, sample_size: int = 200) -> dict[str, Any]:
        """Probes store consistency between SQLite and Qdrant:
        1. Checks total point count equality.
        2. Samples random passage IDs from SQLite and probes their presence in Qdrant.
        """
        sqlite_count = self.get_passage_count()
        qdrant_info = qdrant_store.client.get_collection(qdrant_store.collection_name)
        qdrant_count = qdrant_info.points_count

        count_match = bool(sqlite_count == qdrant_count)

        # Sample passage IDs from SQLite
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT passage_id FROM passages ORDER BY RANDOM() LIMIT ?;",
                (sample_size,),
            )
            sampled_pids = [str(r[0]) for r in cur.fetchall()]

        # Probe in Qdrant
        from prismx.index.qdrant_store import passage_id_to_point_id
        sampled_pt_ids = [passage_id_to_point_id(p) for p in sampled_pids]

        found_in_qdrant = 0
        if sampled_pt_ids:
            try:
                retrieved = qdrant_store.client.retrieve(
                    collection_name=qdrant_store.collection_name,
                    ids=sampled_pt_ids,
                    with_payload=False,
                    with_vectors=False,
                )
                found_in_qdrant = len(retrieved)
            except Exception as e:
                logger.error(f"Consistency probe retrieval failed: {e}")

        probe_match = bool(found_in_qdrant == len(sampled_pids))
        is_consistent = count_match and probe_match

        return {
            "is_consistent": is_consistent,
            "sqlite_count": sqlite_count,
            "qdrant_count": qdrant_count,
            "count_difference": abs(sqlite_count - qdrant_count),
            "sampled_count": len(sampled_pids),
            "sampled_found_in_qdrant": found_in_qdrant,
            "pending_outbox_count": self.get_outbox_pending_count(),
        }

    def get_stats(self) -> dict[str, Any]:
        """Calculates true_avgdl, drift, and version stats."""
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
            "outbox_pending_count": self.get_outbox_pending_count(),
        }
