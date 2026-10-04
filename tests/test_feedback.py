"""Unit Tests for Gate 15 B2 Asynchronous Feedback Storage."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prismx.api.feedback_store import init_feedback_db, record_feedback_async
from scripts.export_feedback_to_eval import export_feedback


def test_feedback_async_recording_and_isolation(tmp_path: Path):
    """Verify feedback writes to isolated DB and never to benchmark stores."""
    test_db = tmp_path / "test_feedback.db"
    fb_id = f"fb_{uuid.uuid4().hex[:8]}"

    # Invariant: Forbidden benchmark stores raise error
    with pytest.raises(ValueError):
        init_feedback_db(Path("data/c100k_raw/text_store_raw.db"))

    # Record feedback asynchronously
    record_feedback_async(
        feedback_id=fb_id,
        query_id="q101",
        query="what is quantum computing",
        passage_id="p999",
        vote=1,
        comment="Very helpful!",
        client_ip="127.0.0.1",
        db_path=test_db,
    )

    assert test_db.exists()
    with sqlite3.connect(str(test_db)) as conn:
        row = conn.execute("SELECT query, passage_id, vote, comment FROM user_feedback WHERE feedback_id = ?", (fb_id,)).fetchone()
        assert row is not None
        assert row[0] == "what is quantum computing"
        assert row[1] == "p999"
        assert row[2] == 1
        assert row[3] == "Very helpful!"


def test_feedback_export_jsonl(tmp_path: Path):
    """Verify exporting feedback to JSONL evaluation draft."""
    test_db = tmp_path / "export_feedback.db"
    out_jsonl = tmp_path / "draft_eval.jsonl"

    record_feedback_async(
        feedback_id="fb_exp_1",
        query_id="q201",
        query="what causes rust",
        passage_id="p500",
        vote=1,
        comment="Direct answer",
        db_path=test_db,
    )

    count = export_feedback(test_db, out_jsonl)
    assert count == 1
    assert out_jsonl.exists()
    content = out_jsonl.read_text(encoding="utf-8")
    assert "what causes rust" in content
    assert "p500" in content
