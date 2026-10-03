"""PRISMX Corpus A (standard-100K) Deterministic Builder and Sanity Reporter."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
from pathlib import Path
import random
import time
from typing import Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def compute_stable_hash(seed: int, docid: str) -> int:
    """Computes a 64-bit integer hash from SHA-256 for deterministic ranking."""
    key = f"{seed}_{docid}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16)

def build_corpus_100k(
    corpus_gz_path: Path | None = None,
    splits_manifest_path: Path | None = None,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    target_size: int = 100000,
    seed: int = 42,
    source_name: str = "Tevatron/msmarco-passage-corpus",
) -> dict[str, Any]:
    corpus_gz_path = corpus_gz_path or (REPO_ROOT / "data" / "raw" / "corpus.jsonl.gz")
    splits_manifest_path = splits_manifest_path or (REPO_ROOT / "data" / "manifests" / "splits_manifest.json")
    output_path = output_path or (REPO_ROOT / "data" / "corpus_100k.jsonl")
    manifest_path = manifest_path or (REPO_ROOT / "data" / "manifests" / "corpus_manifest.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Collect all gold passage IDs from TUNE and TEST splits
    manifest_dir = REPO_ROOT / "data" / "manifests"
    with open(manifest_dir / "split_tune.json", "r", encoding="utf-8") as f:
        tune_data = json.load(f)
    with open(manifest_dir / "split_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    gold_set: set[str] = set()
    for rec in tune_data + test_data:
        for docid in rec["gold_passage_ids"]:
            gold_set.add(str(docid).strip())

    print(f"Targeting {target_size} passages. Loaded {len(gold_set)} gold passage IDs.")

    # Heap capacity: we need (target_size - len(gold_set)) plus safety margin for deduplication
    needed_filler = target_size - len(gold_set)
    heap_capacity = needed_filler + 6000

    gold_passages: dict[str, str] = {}
    # Max-heap to store the lowest hash values: (-hash_val, docid, text)
    filler_heap: list[tuple[int, str, str]] = []

    print(f"Streaming through {corpus_gz_path} in a single pass...")
    t0 = time.time()
    total_scanned = 0

    with gzip.open(corpus_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            total_scanned += 1
            if total_scanned % 1000000 == 0:
                print(f"  Scanned {total_scanned:,} passages ({time.time() - t0:.1f}s)...")

            item = json.loads(line)
            docid = str(item.get("docid") or item.get("_id")).strip()
            text = (item.get("text") or "").strip()

            if docid in gold_set:
                gold_passages[docid] = text
            else:
                h_val = compute_stable_hash(seed, docid)
                if len(filler_heap) < heap_capacity:
                    heapq.heappush(filler_heap, (-h_val, docid, text))
                elif h_val < -filler_heap[0][0]:
                    heapq.heapreplace(filler_heap, (-h_val, docid, text))

    print(f"Streaming complete in {time.time() - t0:.1f}s. Scanned {total_scanned:,} passages.")
    print(f"Retrieved {len(gold_passages)} / {len(gold_set)} gold passages.")
    missing_gold = gold_set - set(gold_passages.keys())
    if missing_gold:
        print(f"WARNING: {len(missing_gold)} gold passages were not found in corpus file!")

    # 2. Extract filler passages and sort by hash ascending
    filler_candidates = [(-neg_h, docid, text) for neg_h, docid, text in filler_heap]
    filler_candidates.sort(key=lambda x: x[0])

    # 3. Deduplicate texts (preserve gold passage copy if a duplicate exists)
    text_to_passage: dict[str, dict[str, Any]] = {}
    duplicates_removed = 0

    # First add all gold passages
    for docid, text in gold_passages.items():
        if text in text_to_passage:
            duplicates_removed += 1
        text_to_passage[text] = {
            "passage_id": docid,
            "text": text,
            "source": source_name,
            "_is_gold": True,
            "_hash": 0,
        }

    # Then add filler passages if text not seen
    for h_val, docid, text in filler_candidates:
        if text in text_to_passage:
            duplicates_removed += 1
            continue
        text_to_passage[text] = {
            "passage_id": docid,
            "text": text,
            "source": source_name,
            "_is_gold": False,
            "_hash": h_val,
        }

    print(f"Removed {duplicates_removed} duplicate texts. Total unique passages: {len(text_to_passage)}")

    # 4. Trim filler by hash rank to exactly target_size
    all_passages = list(text_to_passage.values())
    gold_final = [p for p in all_passages if p["_is_gold"]]
    filler_final = [p for p in all_passages if not p["_is_gold"]]
    filler_final.sort(key=lambda x: x["_hash"])

    filler_needed = target_size - len(gold_final)
    selected_filler = filler_final[:filler_needed]

    final_corpus = gold_final + selected_filler
    assert len(final_corpus) == target_size, f"Corpus size {len(final_corpus)} != {target_size}"

    # Deterministic final ordering by passage_id (never leak gold/filler status)
    final_corpus.sort(key=lambda x: x["passage_id"])

    # Clean internal flags before persisting
    cleaned_corpus = [
        {
            "passage_id": p["passage_id"],
            "text": p["text"],
            "source": p["source"],
        }
        for p in final_corpus
    ]

    # Write out Corpus A
    with open(output_path, "w", encoding="utf-8") as f:
        for p in cleaned_corpus:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    corpus_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    gold_ratio = len(gold_final) / target_size

    # 5. Sanity Analysis
    lengths_char = [len(p["text"]) for p in cleaned_corpus]
    lengths_words = [len(p["text"].split()) for p in cleaned_corpus]

    def dist_stats(arr: list[int]) -> dict[str, float]:
        np_arr = np.array(arr)
        return {
            "min": int(np.min(np_arr)),
            "p25": float(np.percentile(np_arr, 25)),
            "p50": float(np.percentile(np_arr, 50)),
            "p75": float(np.percentile(np_arr, 75)),
            "p95": float(np.percentile(np_arr, 95)),
            "max": int(np.max(np_arr)),
            "mean": float(np.mean(np_arr)),
        }

    # 10 random query-gold pairs
    rng = random.Random(seed)
    sampled_eval_records = rng.sample(tune_data, min(10, len(tune_data)))
    sample_pairs = []
    gold_id_to_text = {p["passage_id"]: p["text"] for p in cleaned_corpus if p["passage_id"] in gold_set}
    
    verbatim_count = 0
    total_eval_queries = len(tune_data)
    for q_rec in tune_data:
        q_text = q_rec["query"].lower().strip()
        has_verbatim = False
        for g_id in q_rec["gold_passage_ids"]:
            g_text = gold_id_to_text.get(g_id, "").lower()
            if q_text in g_text:
                has_verbatim = True
                break
        if has_verbatim:
            verbatim_count += 1

    verbatim_pct = (verbatim_count / total_eval_queries) * 100 if total_eval_queries else 0.0

    for rec in sampled_eval_records:
        g_id = rec["gold_passage_ids"][0]
        sample_pairs.append({
            "query_id": rec["query_id"],
            "query": rec["query"],
            "gold_passage_id": g_id,
            "gold_passage_text": gold_id_to_text.get(g_id, "N/A")[:120] + "...",
        })

    # Qrels per query distribution
    qrels_counts = [len(r["gold_passage_ids"]) for r in tune_data + test_data]

    # Run BM25s on TUNE split as INFORMATION
    print("Evaluating bm25s baseline on TUNE split (informative baseline)...")
    import bm25s
    corpus_texts = [p["text"] for p in cleaned_corpus]
    passage_id_to_idx = {p["passage_id"]: idx for idx, p in enumerate(cleaned_corpus)}
    
    retriever = bm25s.BM25(k1=1.2, b=0.75)
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever.index(corpus_tokens)

    tune_queries = [r["query"] for r in tune_data]
    tune_tokens = bm25s.tokenize(tune_queries, stopwords="en")
    results, scores = retriever.retrieve(tune_tokens, k=20)

    # Calculate MRR@10 and Recall@20 on TUNE
    mrr_10_list = []
    recall_20_list = []
    for i, q_rec in enumerate(tune_data):
        gold_ids = set(q_rec["gold_passage_ids"])
        retrieved_ids = [cleaned_corpus[idx]["passage_id"] for idx in results[i]]
        
        # MRR@10
        rr = 0.0
        for rank, pid in enumerate(retrieved_ids[:10], start=1):
            if pid in gold_ids:
                rr = 1.0 / rank
                break
        mrr_10_list.append(rr)

        # Recall@20
        hits = len(gold_ids.intersection(set(retrieved_ids[:20])))
        recall_20_list.append(hits / len(gold_ids) if gold_ids else 0.0)

    bm25s_mrr10 = float(np.mean(mrr_10_list))
    bm25s_recall20 = float(np.mean(recall_20_list))
    print(f"BM25s TUNE Information: MRR@10 = {bm25s_mrr10:.4f}, Recall@20 = {bm25s_recall20:.4f}")

    manifest = {
        "dataset_name": source_name,
        "corpus_name": "Corpus A (standard-100K)",
        "total_passages": target_size,
        "gold_passages_count": len(gold_final),
        "filler_passages_count": len(selected_filler),
        "gold_to_total_ratio": round(gold_ratio, 6),
        "duplicates_removed": duplicates_removed,
        "seed": seed,
        "sha256": corpus_sha256,
        "length_distribution_chars": dist_stats(lengths_char),
        "length_distribution_words": dist_stats(lengths_words),
        "verbatim_query_in_gold_pct": round(verbatim_pct, 2),
        "qrels_per_query_distribution": {
            "mean": float(np.mean(qrels_counts)),
            "min": int(np.min(qrels_counts)),
            "max": int(np.max(qrels_counts)),
        },
        "informative_bm25s_tune": {
            "mrr_at_10": round(bm25s_mrr10, 4),
            "recall_at_20": round(bm25s_recall20, 4),
            "note": "Informational only on 100K sub-corpus; not full 8.8M MS MARCO collection.",
        },
        "sample_pairs": sample_pairs,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest

if __name__ == "__main__":
    m = build_corpus_100k()
    print("Corpus A build complete. Manifest written to data/manifests/corpus_manifest.json")
