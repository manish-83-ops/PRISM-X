"""PRISMX Gate 5: Hard-Distractor Stress Test Runner.

Adheres strictly to ADR-015:
- Separate Qdrant collection 'c100k_hard' (original 'prismx_corpus' untouched).
- Hard distractors mined from MS MARCO pool (excluding 100k and qrel-positives).
- Single-run evaluation on BENCH queries with paired bootstrap.
- RAGAS answer-based evaluation on frozen 25 queries for dense vs hybrid.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
from typing import Any
import numpy as np
import torch
from qdrant_client import QdrantClient, models

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prismx.config import load_config
from prismx.index.encoder import DenseEncoder
from prismx.index.lexical import BM25Tokenizer
from prismx.index.qdrant_store import passage_id_to_point_id
from prismx.retrieve.dense import DenseRetriever
from prismx.retrieve.hybrid import HybridRetriever
from prismx.retrieve.rerank import CrossEncoderReranker
from prismx.retrieve.fusion import FusionEngine
from prismx.eval.metrics import hit_at_1, mrr_at_k, ndcg_at_k, recall_at_k
from prismx.eval.bootstrap import bootstrap_ci, paired_bootstrap_difference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stress_test")

REPO_ROOT = Path(__file__).resolve().parent.parent
STRESS_DIR = REPO_ROOT / "results" / "stress_test"
STRESS_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_HARD = "c100k_hard"
SQLITE_HARD_PATH = REPO_ROOT / "data" / "text_store_hard.db"


def mine_hard_distractors(
    encoder: DenseEncoder,
    queries: list[dict[str, Any]],
    k_per_query: int = 20,
    max_candidate_scan: int = 200000,
    target_candidates: int = 6000,
) -> list[dict[str, Any]]:
    """Mine top-k dense-nearest passages from MS MARCO pool excluding 100K and gold IDs."""
    logger.info("Mining hard distractors from MS MARCO raw corpus...")

    # Load 100k IDs to exclude
    corpus_ids = set()
    with open(REPO_ROOT / "data" / "corpus_100k.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus_ids.add(str(item.get("docid") or item.get("passage_id") or item.get("_id")).strip())

    # Load dev qrel gold IDs to exclude
    gold_ids = set()
    with open(REPO_ROOT / "data" / "raw" / "dev_qrels.tsv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1] != "corpus-id":
                gold_ids.add(str(parts[1]).strip())

    logger.info(f"Exclusion sets: {len(corpus_ids)} 100k IDs, {len(gold_ids)} dev qrel gold IDs.")

    # Extract non-stopword query keywords
    stopwords = {
        "what", "when", "where", "which", "who", "why", "how", "is", "are", "was",
        "were", "the", "a", "an", "in", "on", "of", "for", "to", "and", "do", "does",
        "did", "can", "could", "would", "should", "with", "from", "that", "this"
    }
    query_terms = set()
    for q in queries:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", q["query"].lower())
        for w in words:
            if len(w) > 3 and w not in stopwords:
                query_terms.add(w)

    logger.info(f"Extracted {len(query_terms)} query terms across {len(queries)} queries.")

    # Stream candidates from MS MARCO raw corpus
    candidates: list[tuple[str, str]] = []
    with gzip.open(REPO_ROOT / "data" / "raw" / "corpus.jsonl.gz", "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_candidate_scan:
                break
            item = json.loads(line)
            docid = str(item.get("docid") or item.get("_id")).strip()
            if docid in corpus_ids or docid in gold_ids:
                continue
            text = (item.get("text") or "").strip()
            # Fast keyword filter
            text_lower = text.lower()
            hit_terms = sum(1 for term in query_terms if term in text_lower)
            if hit_terms >= 2:
                candidates.append((docid, text))
            if len(candidates) >= target_candidates:
                break

    logger.info(f"Scanned candidate pool: {len(candidates)} passages.")

    # Encode queries and candidates
    q_texts = [q["query"] for q in queries]
    q_embs = encoder.encode_queries(q_texts)

    c_texts = [c[1] for c in candidates]
    logger.info(f"Encoding {len(c_texts)} candidates with BAAI/bge-small-en-v1.5...")
    c_embs = encoder.model.encode(
        c_texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Compute similarity matrix Q x C
    sim_matrix = np.dot(q_embs, c_embs.T)  # [n_queries, n_candidates]

    chosen_indices = set()
    for q_idx in range(len(queries)):
        top_k_idx = np.argsort(sim_matrix[q_idx])[::-1][:k_per_query]
        chosen_indices.update(top_k_idx)

    unique_distractors = []
    for idx in chosen_indices:
        docid, text = candidates[idx]
        unique_distractors.append({
            "passage_id": docid,
            "text": text,
            "category": "stress-distractor",
            "source": "msmarco-distractor",
        })

    logger.info(f"Selected {len(unique_distractors)} unique hard distractors (top-{k_per_query} per query).")
    return unique_distractors


def setup_c100k_hard_collection(
    client: QdrantClient,
    distractors: list[dict[str, Any]],
    encoder: DenseEncoder,
    tokenizer: BM25Tokenizer,
) -> None:
    """Clones prismx_corpus into c100k_hard and inserts hard distractors."""
    logger.info(f"Setting up isolated collection '{COLLECTION_HARD}'...")

    # Get configuration from original prismx_corpus
    orig_info = client.get_collection("prismx_corpus")
    v_config = orig_info.config.params.vectors
    sv_config = orig_info.config.params.sparse_vectors

    # Check if collection already completely setup with hard distractors
    if client.collection_exists(COLLECTION_HARD):
        cnt = client.count(collection_name=COLLECTION_HARD, exact=True).count
        if cnt >= 102887:
            logger.info(f"'{COLLECTION_HARD}' already has {cnt} points with distractors, skipping setup.")
            return

    # Check if 100,000 base points already exist in c100k_hard
    already_cloned = False
    if client.collection_exists(COLLECTION_HARD):
        cnt = client.count(collection_name=COLLECTION_HARD, exact=True).count
        if cnt >= 100000:
            logger.info(f"'{COLLECTION_HARD}' already has {cnt} points, skipping base point re-cloning.")
            already_cloned = True
        else:
            client.delete_collection(COLLECTION_HARD)

    if not already_cloned:
        client.create_collection(
            collection_name=COLLECTION_HARD,
            vectors_config=v_config,
            sparse_vectors_config=sv_config,
        )

        client.create_payload_index(
            collection_name=COLLECTION_HARD,
            field_name="category",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        client.create_payload_index(
            collection_name=COLLECTION_HARD,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

        # Scroll and copy all points from prismx_corpus
        logger.info("Copying 100,000 points from 'prismx_corpus' to 'c100k_hard'...")
        offset = None
        copied_count = 0
        t0 = time.perf_counter()

        while True:
            pts, next_offset = client.scroll(
                collection_name="prismx_corpus",
                limit=2500,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not pts:
                break

            records = [
                models.PointStruct(
                    id=p.id,
                    payload=p.payload,
                    vector=p.vector,
                )
                for p in pts
            ]
            client.upsert(collection_name=COLLECTION_HARD, points=records, wait=True)
            copied_count += len(pts)
            offset = next_offset
            if offset is None:
                break

        logger.info(f"Copied {copied_count} base points in {time.perf_counter() - t0:.1f}s.")

    # Copy SQLite text store if not already done
    if not SQLITE_HARD_PATH.exists():
        logger.info(f"Copying text store to {SQLITE_HARD_PATH}...")
        shutil.copy2(REPO_ROOT / "data" / "text_store.db", SQLITE_HARD_PATH)

    # Prepare and upsert distractors
    logger.info(f"Encoding and upserting {len(distractors)} hard distractors...")
    d_texts = [d["text"] for d in distractors]
    d_embs = encoder.model.encode(d_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    distractor_points = []
    sqlite_conn = sqlite3.connect(str(SQLITE_HARD_PATH))
    cursor = sqlite_conn.cursor()

    for idx, d in enumerate(distractors):
        pid = d["passage_id"]
        text = d["text"]
        category = d["category"]
        source = d["source"]

        pt_id = passage_id_to_point_id(pid)
        dense_vec = d_embs[idx].tolist()

        sparse_indices, sparse_values = tokenizer.compute_doc_sparse_vector(text, avgdl_ref=33.6145)

        point = models.PointStruct(
            id=pt_id,
            payload={"passage_id": pid, "category": category, "source": source},
            vector={
                "dense": dense_vec,
                "bm25": models.SparseVector(indices=sparse_indices, values=sparse_values),
            },
        )
        distractor_points.append(point)

        cursor.execute(
            "INSERT OR REPLACE INTO passages (passage_id, text, category, source) VALUES (?, ?, ?, ?)",
            (pid, text, category, source),
        )

    sqlite_conn.commit()
    sqlite_conn.close()

    # Batch upsert distractors to Qdrant
    batch_size = 500
    for b in range(0, len(distractor_points), batch_size):
        chunk = distractor_points[b : b + batch_size]
        client.upsert(collection_name=COLLECTION_HARD, points=chunk, wait=True)

    final_count = client.count(collection_name=COLLECTION_HARD, exact=True).count
    logger.info(f"Successfully initialized '{COLLECTION_HARD}' with {final_count} total points.")


def evaluate_bench_on_hard_collection(
    client: QdrantClient,
    encoder: DenseEncoder,
    tokenizer: BM25Tokenizer,
    reranker: CrossEncoderReranker,
    bench_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluates Dense, Hybrid, and Hybrid+Rerank(K=10) on c100k_hard."""
    logger.info(f"Evaluating {len(bench_queries)} BENCH queries on '{COLLECTION_HARD}'...")

    sqlite_conn = sqlite3.connect(str(SQLITE_HARD_PATH))
    sqlite_conn.row_factory = sqlite3.Row

    # Pre-encode all queries
    q_texts = [q["query"] for q in bench_queries]
    q_embs = encoder.encode_queries(q_texts)

    results = {
        "dense": {"hit1": [], "mrr10": [], "ndcg5": [], "ndcg10": [], "r10": []},
        "hybrid": {"hit1": [], "mrr10": [], "ndcg5": [], "ndcg10": [], "r10": []},
        "hybrid_rerank": {"hit1": [], "mrr10": [], "ndcg5": [], "ndcg10": [], "r10": []},
    }

    retrieval_records = {"dense": {}, "hybrid": {}, "hybrid_rerank": {}}

    for idx, q_item in enumerate(bench_queries):
        qid = str(q_item["query_id"])
        query = q_item["query"]
        gold_ids = set(str(g) for g in q_item["gold_passage_ids"])
        q_vec = q_embs[idx].tolist()

        # 1. Dense retrieval (limit=50)
        dense_resp = client.query_points(
            collection_name=COLLECTION_HARD,
            query=q_vec,
            using="dense",
            limit=50,
            with_payload=True,
            with_vectors=False,
        )
        dense_pts = dense_resp.points
        dense_pids = [str(p.payload["passage_id"]) for p in dense_pts]
        dense_scores = {str(p.payload["passage_id"]): float(p.score) for p in dense_pts}

        # 2. Sparse retrieval (limit=50)
        s_ind, s_val = tokenizer.compute_query_sparse_vector(query)
        sparse_pts = client.query_points(
            collection_name=COLLECTION_HARD,
            query=models.SparseVector(indices=s_ind, values=s_val),
            using="bm25",
            limit=50,
            with_payload=True,
            with_vectors=False,
        ).points
        # 3. Hybrid fusion (alpha=0.8, minmax)
        dense_candidates = [
            {"passage_id": str(p.payload["passage_id"]), "dense_score": float(p.score), "dense_rank": r, "category": p.payload.get("category"), "source": p.payload.get("source")}
            for r, p in enumerate(dense_pts, start=1)
        ]
        sparse_candidates = [
            {"passage_id": str(p.payload.get("passage_id", p.id)), "bm25_score": float(p.score), "bm25_rank": r, "category": p.payload.get("category"), "source": p.payload.get("source")}
            for r, p in enumerate(sparse_pts, start=1)
        ]
        fused = FusionEngine.fuse_weighted(dense_candidates, sparse_candidates, alpha=0.8, norm_method="minmax")
        hybrid_pids = [c["passage_id"] for c in fused]
        fused_scores = {c["passage_id"]: c["score"] for c in fused}

        # 4. Rerank K=10 candidates
        rerank_candidates = []
        cands_to_fetch = hybrid_pids[:10]
        cur = sqlite_conn.cursor()
        placeholders = ",".join(["?"] * len(cands_to_fetch))
        cur.execute(f"SELECT passage_id, text FROM passages WHERE passage_id IN ({placeholders})", cands_to_fetch)
        text_map = {row["passage_id"]: row["text"] for row in cur.fetchall()}

        for pid in cands_to_fetch:
            rerank_candidates.append({
                "passage_id": pid,
                "text": text_map.get(pid, ""),
                "score": fused_scores.get(pid, 0.0),
            })

        reranked_pool, _, _ = reranker.rerank(
            query=query,
            candidates=rerank_candidates,
            top_k=10,
            max_length=128,
            deadline_ms=200.0,
            batch_size=5,
        )
        rerank_pids = [r["passage_id"] for r in reranked_pool]
        # Append remaining fused candidates beyond 10
        rerank_pids.extend(hybrid_pids[10:])

        # Save retrieval records
        retrieval_records["dense"][qid] = dense_pids[:10]
        retrieval_records["hybrid"][qid] = hybrid_pids[:10]
        retrieval_records["hybrid_rerank"][qid] = rerank_pids[:10]

        # Compute metrics
        for mode, pids in [("dense", dense_pids), ("hybrid", hybrid_pids), ("hybrid_rerank", rerank_pids)]:
            top10 = pids[:10]
            top5 = pids[:5]
            results[mode]["hit1"].append(hit_at_1(top10, gold_ids))
            results[mode]["mrr10"].append(mrr_at_k(top10, gold_ids, 10))
            results[mode]["ndcg5"].append(ndcg_at_k(top5, gold_ids, 5))
            results[mode]["ndcg10"].append(ndcg_at_k(top10, gold_ids, 10))
            results[mode]["r10"].append(recall_at_k(top10, gold_ids, 10))

    sqlite_conn.close()

    # Compute bootstrap summary and deltas
    summary = {}
    for mode in ["dense", "hybrid", "hybrid_rerank"]:
        summary[mode] = {
            m: bootstrap_ci(results[mode][m], n_resamples=10000)
            for m in ["hit1", "mrr10", "ndcg5", "ndcg10", "r10"]
        }

    deltas = {
        "hybrid_vs_dense": {
            m: paired_bootstrap_difference(results["dense"][m], results["hybrid"][m], n_resamples=10000)
            for m in ["hit1", "mrr10", "ndcg5", "ndcg10", "r10"]
        },
        "rerank_vs_hybrid": {
            m: paired_bootstrap_difference(results["hybrid"][m], results["hybrid_rerank"][m], n_resamples=10000)
            for m in ["hit1", "mrr10", "ndcg5", "ndcg10", "r10"]
        },
        "rerank_vs_dense": {
            m: paired_bootstrap_difference(results["dense"][m], results["hybrid_rerank"][m], n_resamples=10000)
            for m in ["hit1", "mrr10", "ndcg5", "ndcg10", "r10"]
        },
    }

    return {
        "summary": summary,
        "deltas": deltas,
        "retrieval_records": retrieval_records,
    }


def run_stress_test_ragas(
    frozen_queries: list[dict[str, Any]],
    retrievals: dict[str, Any],
) -> dict[str, Any]:
    """Runs RAGAS answer-based CP/CR for Dense vs Hybrid on the frozen 25 queries."""
    logger.info("Running RAGAS evaluation on 25 frozen queries for 'c100k_hard'...")
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment, skipping RAGAS on c100k_hard.")
        return {"status": "skipped", "reason": "No GROQ_API_KEY"}

    groq_client = Groq(api_key=api_key)

    sqlite_conn = sqlite3.connect(str(SQLITE_HARD_PATH))
    sqlite_conn.row_factory = sqlite3.Row

    # Evaluate dense vs hybrid
    ragas_results = {
        "dense": {"cp": [], "cr": []},
        "hybrid": {"cp": [], "cr": []},
    }

    # Import helper evaluators
    from scripts.run_gate4b_ragas_frozen25 import eval_context_precision, eval_context_recall

    total_tokens = 0
    t0 = time.perf_counter()

    for idx, item in enumerate(frozen_queries, start=1):
        qid = str(item["query_id"])
        query = item["query"]
        ref_ans = item["reference_answer"]

        # Get top-5 passage texts for dense and hybrid
        for mode in ["dense", "hybrid"]:
            pids = retrievals[mode][qid][:5]
            placeholders = ",".join(["?"] * len(pids))
            cur = sqlite_conn.cursor()
            cur.execute(f"SELECT passage_id, text FROM passages WHERE passage_id IN ({placeholders})", pids)
            t_map = {row["passage_id"]: row["text"] for row in cur.fetchall()}
            contexts = [t_map.get(p, "") for p in pids]

            cp, tok_cp = eval_context_precision(groq_client, query, contexts, ref_ans)
            cr, tok_cr = eval_context_recall(groq_client, query, contexts, ref_ans)
            total_tokens += (tok_cp + tok_cr)

            ragas_results[mode]["cp"].append(cp)
            ragas_results[mode]["cr"].append(cr)

        logger.info(f"[{idx}/25] QID {qid} scored (Dense CP={ragas_results['dense']['cp'][-1]:.4f}, Hybrid CP={ragas_results['hybrid']['cp'][-1]:.4f})")

    sqlite_conn.close()

    elapsed = time.perf_counter() - t0
    summary = {
        "dense": {
            "context_precision": bootstrap_ci(ragas_results["dense"]["cp"]),
            "context_recall": bootstrap_ci(ragas_results["dense"]["cr"]),
        },
        "hybrid": {
            "context_precision": bootstrap_ci(ragas_results["hybrid"]["cp"]),
            "context_recall": bootstrap_ci(ragas_results["hybrid"]["cr"]),
        },
        "difference_hybrid_vs_dense": {
            "context_precision": paired_bootstrap_difference(ragas_results["dense"]["cp"], ragas_results["hybrid"]["cp"]),
            "context_recall": paired_bootstrap_difference(ragas_results["dense"]["cr"], ragas_results["hybrid"]["cr"]),
        },
        "telemetry": {
            "n_queries": len(frozen_queries),
            "evaluator_model": "allam-2-7b",
            "total_tokens": total_tokens,
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    return summary


def main():
    logger.info("====================================================================")
    logger.info("GATE 5: HARD-DISTRACTOR STRESS TEST (ADR-015)")
    logger.info("====================================================================")

    cfg = load_config()
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"], grpc_port=cfg["qdrant"]["grpc_port"], prefer_grpc=True)

    encoder = DenseEncoder(model_name=cfg["encoder"]["model_name"], embedding_dim=cfg["encoder"]["embedding_dim"], max_seq_length=128, torch_threads=8)
    tokenizer = BM25Tokenizer(k1=cfg["lexical"]["k1"], b=cfg["lexical"]["b"], use_stemming=cfg["lexical"].get("stemming", False))
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", torch_threads=8)

    # 1. Load evaluation queries
    with open(REPO_ROOT / "data" / "manifests" / "split_bench.json") as f:
        bench_queries = json.load(f)
    with open(REPO_ROOT / "data" / "manifests" / "split_tune.json") as f:
        tune_queries = json.load(f)[:150]

    all_eval_queries = bench_queries + tune_queries

    # 2. Mine hard distractors and setup collection if not already completed
    need_setup = True
    distractors_count = 2887
    if client.collection_exists(COLLECTION_HARD):
        cnt = client.count(collection_name=COLLECTION_HARD, exact=True).count
        if cnt >= 102887:
            need_setup = False
            logger.info(f"Collection '{COLLECTION_HARD}' already has {cnt} points, skipping mining and setup.")

    if need_setup:
        distractors = mine_hard_distractors(encoder, all_eval_queries, k_per_query=20)
        distractors_count = len(distractors)
        setup_c100k_hard_collection(client, distractors, encoder, tokenizer)

    summary_path = STRESS_DIR / "stress_test_summary.json"
    retrievals_path = STRESS_DIR / "stress_test_retrievals.json"

    if summary_path.exists() and retrievals_path.exists():
        logger.info(f"Loading cached stress test retrieval records from {retrievals_path}...")
        with open(retrievals_path, "r", encoding="utf-8") as f:
            retrieval_records = json.load(f)
    else:
        # 3. Evaluate BENCH queries on c100k_hard
        eval_results = evaluate_bench_on_hard_collection(client, encoder, tokenizer, reranker, bench_queries)
        retrieval_records = eval_results["retrieval_records"]

        # Save retrieval records
        with open(retrievals_path, "w", encoding="utf-8") as f:
            json.dump(retrieval_records, f, indent=2)

        # 4. Save quality evaluation results
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "label": "Stress test (hard distractors, unlabeled neighbors may be valid answers; ID metrics are pessimistic)",
                "collection": COLLECTION_HARD,
                "points_count": client.count(collection_name=COLLECTION_HARD, exact=True).count,
                "distractors_added": distractors_count,
                "results": eval_results["summary"],
                "paired_deltas": eval_results["deltas"],
            }, f, indent=2)

        logger.info(f"Saved stress test benchmark summary to {summary_path}")

    # 6. RAGAS evaluation on frozen 25 queries
    with open(REPO_ROOT / "data" / "manifests" / "frozen_ragas_bench_queries.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        frozen_queries = data.get("frozen_queries_n25", data)

    ragas_results = run_stress_test_ragas(frozen_queries, retrieval_records)

    ragas_path = STRESS_DIR / "stress_test_ragas.json"
    with open(ragas_path, "w", encoding="utf-8") as f:
        json.dump({
            "label": "Stress test RAGAS (hard distractors, unlabeled neighbors may be valid answers; ID metrics are pessimistic)",
            "collection": COLLECTION_HARD,
            "ragas": ragas_results,
        }, f, indent=2)

    logger.info(f"Saved stress test RAGAS summary to {ragas_path}")
    logger.info("====================================================================")
    logger.info("GATE 5 HARD-DISTRACTOR STRESS TEST COMPLETE!")
    logger.info("====================================================================")


if __name__ == "__main__":
    main()
