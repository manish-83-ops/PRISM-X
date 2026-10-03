"""Generate recorded API responses from real results files for offline UI dev."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
RECORDED = ROOT / "frontend" / "public" / "recorded"
PUB_DATA = ROOT / "frontend" / "public" / "data"

RECORDED.mkdir(parents=True, exist_ok=True)
PUB_DATA.mkdir(parents=True, exist_ok=True)

# 1. Load bench queries
bench = json.loads((DATA / "manifests" / "split_bench.json").read_text())
bench_by_id = {q["query_id"]: q for q in bench}

# 2. Load dense + hybrid retrievals
dense_ret = json.loads((RESULTS / "phase1" / "bench_dense_retrievals.json").read_text())
hybrid_ret = json.loads((RESULTS / "phase2" / "bench_hybrid_retrievals.json").read_text())

# 3. Build fake SearchResponse-shaped recorded responses for 10 sample queries
sample_indices = [0, 1, 2, 3, 4, 7, 10, 15, 20, 25]
selected_queries = []

for idx in sample_indices:
    if idx >= len(bench):
        continue
    q = bench[idx]
    qid = q["query_id"]
    query = q["query"]
    gold_pids = q.get("gold_passage_ids", [])
    selected_queries.append({"query_id": qid, "query": query, "gold_passage_ids": gold_pids})

    # Find matching dense retrieval
    d_match = next((r for r in dense_ret if r.get("query_id") == qid), None)
    h_match = next((r for r in hybrid_ret if r.get("query_id") == qid), None)

    for mode, match in [("dense", d_match), ("hybrid", h_match)]:
        if not match:
            continue
        # Build SearchResponse shape
        results = []
        ids = match.get("retrieved_ids", [])
        scores = match.get("retrieved_scores", [])
        texts = match.get("retrieved_texts", [])
        for rank_i, pid in enumerate(ids[:5]):
            results.append({
                "rank": rank_i + 1,
                "passage_id": pid,
                "text": texts[rank_i] if rank_i < len(texts) else f"[Passage {pid} text not available in offline mode]",
                "category": None,
                "source": "msmarco-passage",
                "score": round(scores[rank_i], 4) if rank_i < len(scores) else 0.0,
                "dense_rank": rank_i + 1 if mode == "dense" else (rank_i + 2 if rank_i > 0 else 1),
                "dense_score": round(scores[rank_i], 4) if mode == "dense" else round(scores[rank_i] * 0.9, 4),
                "bm25_rank": None if mode == "dense" else rank_i + 1,
                "bm25_score": None if mode == "dense" else round(scores[rank_i] * 0.3, 4),
            })

        latency = match.get("latency_ms", 45.0)
        response = {
            "query": query,
            "mode": mode,
            "fusion_used": {"method": "weighted", "alpha": 0.8, "rrf_k": 60} if mode == "hybrid" else None,
            "filters_applied": None,
            "index_version": 3,
            "results": results,
            "latency_ms": {
                "encode": round(latency * 0.25, 2),
                "dense": round(latency * 0.35, 2),
                "sparse": round(latency * 0.15, 2) if mode == "hybrid" else 0.0,
                "fusion": round(latency * 0.05, 2) if mode == "hybrid" else 0.0,
                "fetch_text": round(latency * 0.15, 2),
                "total": round(latency, 2),
            },
            "_recorded": True,
        }
        fname = f"search_{mode}_{qid}.json"
        (RECORDED / fname).write_text(json.dumps(response, indent=2))

# 4. Save sample query list
(RECORDED / "sample_queries.json").write_text(json.dumps(selected_queries, indent=2))

# 5. Build recorded meta
meta = {
    "modes": ["dense", "hybrid"],
    "fusion_defaults": {"method": "weighted", "alpha": 0.8, "rrf_k": 60},
    "point_count": 100000,
    "sqlite_count": 100000,
    "index_version": 3,
    "avgdl_ref": 56.43,
    "true_avgdl": 56.43,
    "drift": 0.0,
    "drift_warning": False,
    "inconsistency_count": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_hit_rate": 0.0,
    "categories": [
        "calories-food", "click-use", "tax-state", "cost-average", "symptoms-pain",
        "blood-body", "county-city", "meaning-definition", "water-use", "used-data",
        "new-year", "time-average", "water-cell", "people-health", "states-war",
    ],
    "sources": ["msmarco-passage", "manual"],
    "models": {"dense": "sentence-transformers/all-MiniLM-L6-v2", "lexical": "qdrant_sparse_bm25_idf"},
    "config_hash": "recorded",
    "_recorded": True,
}
(RECORDED / "meta.json").write_text(json.dumps(meta, indent=2))

# 6. Copy results files to public/data/ for the Evaluation page
copy_files = [
    ("phase1/metrics.json", "phase1_metrics.json"),
    ("phase2/metrics.json", "phase2_metrics.json"),
    ("phase2/benchmark_summary.json", "benchmark_summary.json"),
    ("phase1/ragas_eval.json", "phase1_ragas.json"),
    ("phase2/ragas_eval.json", "phase2_ragas.json"),
]
for src, dst in copy_files:
    src_path = RESULTS / src
    if src_path.exists():
        (PUB_DATA / dst).write_text(src_path.read_text())
        print(f"  Copied {src} -> data/{dst}")

# 7. Copy latency CSVs
for csv_name in ["latency_dense.csv", "latency_hybrid_uncached.csv", "latency_hybrid_cached.csv"]:
    for phase_dir in ["phase1", "phase2"]:
        src_path = RESULTS / phase_dir / csv_name
        if src_path.exists():
            dst_name = f"{phase_dir}_{csv_name}" if phase_dir == "phase1" else csv_name
            (PUB_DATA / dst_name).write_text(src_path.read_text())
            print(f"  Copied {phase_dir}/{csv_name} -> data/{dst_name}")

# 8. Copy RAGAS paired checkpoint
ragas_paired = RESULTS / "ragas" / "paired_checkpoint.json"
if ragas_paired.exists():
    (PUB_DATA / "ragas_paired.json").write_text(ragas_paired.read_text())
    print("  Copied ragas/paired_checkpoint.json -> data/ragas_paired.json")

# 9. Copy cluster metadata
cluster = DATA / "manifests" / "cluster_metadata.json"
if cluster.exists():
    (PUB_DATA / "cluster_metadata.json").write_text(cluster.read_text())
    print("  Copied cluster_metadata.json -> data/cluster_metadata.json")

# 10. Copy bench queries
(PUB_DATA / "bench_queries.json").write_text(json.dumps(bench, indent=2))
print(f"  Wrote bench_queries.json ({len(bench)} queries)")

print(f"\nDone. Generated {len(list(RECORDED.glob('*.json')))} recorded files.")
print(f"Synced {len(list(PUB_DATA.glob('*')))} data files.")
