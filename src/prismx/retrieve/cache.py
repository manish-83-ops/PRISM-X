"""PRISMX In-Memory Query Result Cache with Version Stamping and Reverse Index Eviction.
Conforms to ADR-024:
- Version-stamped keys: Key incorporates corpus_version from SQLite meta.
  Invalidation is O(1): incrementing corpus_version makes old keys unmatchable.
  A slow in-flight request started at version V writes only to key_V, preventing cache pollution for version V+1.
- Reverse index: Maps passage_id -> set of cache keys.
  Deletes and updates can evict specific entries containing the affected passage.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import json
import logging
import threading
from typing import Any

logger = logging.getLogger("prismx.cache")


class QueryCache:
    """Thread-safe LRU query cache with SHA-256 deterministic key generation,
    version stamping, and reverse-index passage eviction.
    """

    def __init__(self, maxsize: int = 2000) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._key_to_passages: dict[str, set[str]] = {}
        self._passage_to_keys: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.Lock()
        self.version: int = 1
        self.hits: int = 0
        self.misses: int = 0

    def bump_version(self) -> int:
        """Bump cache version to invalidate queries stamped with older corpus versions."""
        with self._lock:
            self.version += 1
            return self.version

    def make_key(
        self,
        query: str,
        mode: str,
        top_k: int,
        filters: Any = None,
        fusion: Any = None,
        rerank: bool = False,
        rerank_k: int = 20,
        corpus_version: int = 1,
    ) -> str:
        """Generate a deterministic SHA-256 hash key from query, parameters, and corpus_version."""
        key_obj = {
            "query": query.strip().lower(),
            "mode": mode,
            "top_k": top_k,
            "filters": str(filters) if filters is not None else None,
            "fusion": str(fusion) if fusion is not None else None,
            "rerank": bool(rerank),
            "rerank_k": int(rerank_k),
            "corpus_version": int(corpus_version),
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

    def set(self, key: str, value: Any, passage_ids: list[str] | None = None) -> None:
        """Store result in cache with LRU eviction and passage reverse index tracking."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                # Clean old reverse index mapping for this key if re-written
                if key in self._key_to_passages:
                    for pid in self._key_to_passages[key]:
                        self._passage_to_keys[pid].discard(key)

            self._cache[key] = value

            if passage_ids:
                pids_set = {str(p) for p in passage_ids}
                self._key_to_passages[key] = pids_set
                for pid in pids_set:
                    self._passage_to_keys[pid].add(key)

            if len(self._cache) > self.maxsize:
                evicted_key, _ = self._cache.popitem(last=False)
                # Clean up reverse index for evicted LRU entry
                if evicted_key in self._key_to_passages:
                    for pid in self._key_to_passages[evicted_key]:
                        self._passage_to_keys[pid].discard(evicted_key)
                        if not self._passage_to_keys[pid]:
                            del self._passage_to_keys[pid]
                    del self._key_to_passages[evicted_key]

    def evict_passage(self, passage_id: str) -> int:
        """Evicts all cached query results containing the given passage_id via reverse index."""
        pid = str(passage_id)
        with self._lock:
            keys_to_evict = list(self._passage_to_keys.pop(pid, set()))
            evicted_count = 0
            for k in keys_to_evict:
                if k in self._cache:
                    del self._cache[k]
                    evicted_count += 1
                if k in self._key_to_passages:
                    for other_pid in self._key_to_passages[k]:
                        if other_pid != pid:
                            self._passage_to_keys[other_pid].discard(k)
                            if not self._passage_to_keys[other_pid]:
                                del self._passage_to_keys[other_pid]
                    del self._key_to_passages[k]
            logger.info(f"Evicted {evicted_count} cache entries containing passage '{passage_id}'")
            return evicted_count

    def invalidate(self) -> int:
        """Clear all cached entries and reverse indices."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._key_to_passages.clear()
            self._passage_to_keys.clear()
            logger.info(f"Query cache invalidated: {count} entries cleared.")
            return count

    def clear(self) -> int:
        """Alias for invalidate."""
        return self.invalidate()

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
                "indexed_passages_count": len(self._passage_to_keys),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
