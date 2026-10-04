"""
Prometheus Metrics Collector and Text Exposition for PRISMX API (Gate 15 A1).
Thread-safe, lightweight, zero-dependency Prometheus text metric collector.
Tracks:
- Request counters by mode and HTTP status code
- Latency histograms by retrieval stage (encode, dense, sparse, fusion, fetch_text, rerank, total)
- Governor state counters (normal, truncated, skipped_budget)
- Cache hits and misses
- Active corpus points and system uptime
"""

import collections
import threading
import time
from typing import Any

# Standard latency buckets in milliseconds
LATENCY_BUCKETS_MS = (5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 500.0, 1000.0, float("inf"))


class StageHistogram:
    """Thread-safe histogram for measuring stage latencies in milliseconds."""

    def __init__(self, buckets: tuple[float, ...] = LATENCY_BUCKETS_MS):
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.sum = 0.0
        self.total_count = 0
        self._lock = threading.Lock()

    def observe(self, val_ms: float) -> None:
        with self._lock:
            self.total_count += 1
            self.sum += val_ms
            for idx, b in enumerate(self.buckets):
                if val_ms <= b:
                    self.counts[idx] += 1


class PrometheusMetrics:
    """Singleton metrics registry for PRISMX serving layer."""

    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()

        # Request counter: (mode, status_code) -> count
        self.requests_total: dict[tuple[str, int], int] = collections.defaultdict(int)

        # Stage latency histograms
        self.stage_histograms: dict[str, StageHistogram] = {
            "encode": StageHistogram(),
            "dense": StageHistogram(),
            "sparse": StageHistogram(),
            "fusion": StageHistogram(),
            "fetch_text": StageHistogram(),
            "rerank": StageHistogram(),
            "total": StageHistogram(),
        }

        # Governor state counters: state -> count
        self.governor_states: dict[str, int] = collections.defaultdict(int)

        # Cache counters
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def record_request(self, mode: str, status_code: int) -> None:
        with self._lock:
            self.requests_total[(mode, status_code)] += 1

    def record_stage_latency(self, stage: str, duration_ms: float) -> None:
        hist = self.stage_histograms.get(stage)
        if hist:
            hist.observe(duration_ms)

    def record_stages(self, stages: dict[str, float]) -> None:
        for stage, duration_ms in stages.items():
            if duration_ms is not None:
                self.record_stage_latency(stage, float(duration_ms))

    def record_governor_state(self, state: str) -> None:
        with self._lock:
            self.governor_states[state] += 1

    def record_cache_hit(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def export_text(self, corpus_points: int = 100008) -> str:
        """Render metrics in Prometheus Text Format (version 0.0.4)."""
        lines = []

        # 1. Uptime
        uptime = time.time() - self.start_time
        lines.append("# HELP prismx_uptime_seconds Process uptime in seconds.")
        lines.append("# TYPE prismx_uptime_seconds gauge")
        lines.append(f"prismx_uptime_seconds {uptime:.2f}")

        # 2. Corpus Points
        lines.append("# HELP prismx_corpus_points Total number of indexed points in Qdrant.")
        lines.append("# TYPE prismx_corpus_points gauge")
        lines.append(f"prismx_corpus_points {corpus_points}")

        # 3. Requests Total
        lines.append("# HELP prismx_requests_total Total number of HTTP search and retrieval requests.")
        lines.append("# TYPE prismx_requests_total counter")
        with self._lock:
            req_items = list(self.requests_total.items())
        if req_items:
            for (mode, status_code), count in sorted(req_items):
                lines.append(f'prismx_requests_total{{mode="{mode}",status="{status_code}"}} {count}')
        else:
            lines.append('prismx_requests_total{mode="default",status="200"} 0')

        # 4. Cache Counters
        lines.append("# HELP prismx_cache_hits_total Total query cache hits.")
        lines.append("# TYPE prismx_cache_hits_total counter")
        lines.append(f"prismx_cache_hits_total {self.cache_hits}")
        lines.append("# HELP prismx_cache_misses_total Total query cache misses.")
        lines.append("# TYPE prismx_cache_misses_total counter")
        lines.append(f"prismx_cache_misses_total {self.cache_misses}")

        # 5. Governor State
        lines.append("# HELP prismx_governor_state_total Total governor decisions by outcome state.")
        lines.append("# TYPE prismx_governor_state_total counter")
        with self._lock:
            gov_items = list(self.governor_states.items())
        if gov_items:
            for state, count in sorted(gov_items):
                lines.append(f'prismx_governor_state_total{{state="{state}"}} {count}')
        else:
            for s in ("normal", "truncated", "skipped_budget"):
                lines.append(f'prismx_governor_state_total{{state="{s}"}} 0')

        # 6. Stage Latency Histograms
        lines.append("# HELP prismx_latency_ms Histogram of stage latencies in milliseconds.")
        lines.append("# TYPE prismx_latency_ms histogram")
        for stage, hist in self.stage_histograms.items():
            cumulative = 0
            for idx, le in enumerate(hist.buckets):
                cumulative = hist.counts[idx]
                le_str = "+Inf" if le == float("inf") else str(le)
                lines.append(f'prismx_latency_ms_bucket{{stage="{stage}",le="{le_str}"}} {cumulative}')
            lines.append(f'prismx_latency_ms_sum{{stage="{stage}"}} {hist.sum:.4f}')
            lines.append(f'prismx_latency_ms_count{{stage="{stage}"}} {hist.total_count}')

        lines.append("")
        return "\n".join(lines)


# Global singleton instance
metrics_registry = PrometheusMetrics()
