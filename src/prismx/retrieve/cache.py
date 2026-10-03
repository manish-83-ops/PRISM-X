"""PRISMX In-Memory Query Result Cache with Invalidation."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger("prismx.cache")


class QueryCache:
    """Thread-safe LRU query cache with SHA-256 deterministic key generation."""

    def __init__(self, maxsize: int = 2000) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def make_key(
        self,
        query: str,
        mode: str,
        top_k: int,
        filters: Any = None,
        fusion: Any = None,
        rerank: bool = False,
        rerank_k: int = 20,
    ) -> str:
        """Generate a deterministic SHA-256 hash key from query and retrieval parameters."""
        key_obj = {
            "query": query.strip().lower(),
            "mode": mode,
            "top_k": top_k,
            "filters": str(filters) if filters is not None else None,
            "fusion": str(fusion) if fusion is not None else None,
            "rerank": bool(rerank),
            "rerank_k": int(rerank_k),
        }
        canonical_str = json.dumps(key_obj, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        """Retrieve cached result or None if miss."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Store result in cache with LRU eviction."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def invalidate(self) -> int:
        """Clear all cached entries on upsert or delete."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Query cache invalidated: {count} entries cleared.")
            return count

    def stats(self) -> dict[str, Any]:
        """Return cache health and hit rate statistics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
