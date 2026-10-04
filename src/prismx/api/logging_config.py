"""
Asynchronous Structured JSON Logging for PRISMX Serving Layer (Gate 15 A1).
Uses Python's logging.handlers.QueueHandler and QueueListener to dispatch
all request and application logs off the main request event loop.
Ensures ZERO synchronous per-request disk writes on the request path.
"""

import json
import logging
import logging.handlers
import queue
import sys
import time
from typing import Any, Optional


class JSONLogFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects with ISO timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject extra attributes from record if present
        for attr in ("request_id", "method", "path", "status_code", "duration_ms", "client_ip", "mode"):
            val = getattr(record, attr, None)
            if val is not None:
                log_obj[attr] = val

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log_obj["exception"] = record.exc_text

        return json.dumps(log_obj, ensure_ascii=False)


_log_queue: queue.Queue = queue.Queue(maxsize=10000)
_queue_listener: Optional[logging.handlers.QueueListener] = None


def setup_async_json_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Initialize async queue logging for the prismx server."""
    global _queue_listener

    formatter = JSONLogFormatter()

    # Destination handler (console or file) running in background thread
    dest_handler = logging.FileHandler(log_file, encoding="utf-8") if log_file else logging.StreamHandler(sys.stdout)
    dest_handler.setFormatter(formatter)

    # QueueListener drains _log_queue in a background thread
    if _queue_listener is not None:
        try:
            _queue_listener.stop()
        except Exception:
            pass

    _queue_listener = logging.handlers.QueueListener(_log_queue, dest_handler, respect_handler_level=True)
    _queue_listener.start()

    # QueueHandler intercepts log calls in workers and puts them in _log_queue
    queue_handler = logging.handlers.QueueHandler(_log_queue)

    logger = logging.getLogger("prismx")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(queue_handler)
    logger.propagate = False

    return logger


def shutdown_async_logging() -> None:
    """Stop the background queue listener during server shutdown."""
    global _queue_listener
    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None
