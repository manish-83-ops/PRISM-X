import sys
sys.path.insert(0, 'src')
import time
import json
import numpy as np
from prismx.index.text_store import TextStore
from prismx.schemas import SearchResponse, SearchResultItem, LatencyBreakdown

store = TextStore('data/c100k_raw/text_store_raw.db')
times = []
pids = [str(i) for i in range(100, 110)]

# warm up
for _ in range(10):
    store.get_passages_by_ids(pids)

for _ in range(100):
    t0 = time.perf_counter()
    m = store.get_passages_by_ids(pids)
    items = [
        SearchResultItem(
            rank=i + 1,
            passage_id=pids[i],
            text=m.get(pids[i], {}).get("text", ""),
            score=0.9,
        )
        for i in range(len(pids))
    ]
    resp = SearchResponse(
        query="test",
        mode="hybrid",
        index_version=1,
        results=items,
        latency_ms=LatencyBreakdown(total=1.0),
    )
    _ = resp.model_dump_json()
    times.append((time.perf_counter() - t0) * 1000.0)

p50 = float(np.percentile(times, 50))
p95 = float(np.percentile(times, 95))
p99 = float(np.percentile(times, 99))
max_t = float(np.max(times))
print(f"Fetch + Serialize: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms, max={max_t:.2f}ms")
