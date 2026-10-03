"""Extract and build c100k_raw dataset from microsoft/ms_marco v2.1 validation split.
In strict accordance with ADR-016:
- Sample queries with seed 42
- Ingest candidate passages from sample['passages']
- Deduplicate by normalized text
- Stop when unique passages >= 100,000
- Draw disjoint TUNE (500) and BENCH (100) queries
- Log positive-to-total ratio, duplicate stats, and per-query_type counts
"""

import json
import re
import time
from collections import Counter
from pathlib import Path
from datasets import load_dataset


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def extract_c100k_raw():
    out_dir = Path("data/c100k_raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading microsoft/ms_marco v2.1 validation split...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)

    unique_passages = {}  # norm_text -> dict(passage_id, text, url, category, originating_query_ids)
    all_sampled_queries = []
    
    total_candidates_seen = 0
    duplicate_candidates_count = 0
    query_count = 0
    t0 = time.time()

    print("Sampling queries and extracting candidate passages...")
    for item in ds:
        query_count += 1
        qid = item["query_id"]
        qtext = item["query"]
        qtype = item["query_type"]
        answers = item.get("answers", [])
        well_formed = item.get("wellFormedAnswers", [])

        passages_dict = item["passages"]
        p_texts = passages_dict["passage_text"]
        p_urls = passages_dict["url"]
        p_selected = passages_dict["is_selected"]

        gold_pids = []
        candidate_pids = []

        for p_idx, (text, url, is_sel) in enumerate(zip(p_texts, p_urls, p_selected)):
            total_candidates_seen += 1
            norm = normalize_text(text)
            
            if norm in unique_passages:
                duplicate_candidates_count += 1
                entry = unique_passages[norm]
                if qid not in entry["originating_query_ids"]:
                    entry["originating_query_ids"].append(qid)
                pid = entry["passage_id"]
            else:
                pid = f"raw_{len(unique_passages) + 1}"
                entry = {
                    "passage_id": pid,
                    "text": text,
                    "url": url,
                    "category": qtype,
                    "originating_query_ids": [qid]
                }
                unique_passages[norm] = entry

            candidate_pids.append(pid)
            if is_sel == 1:
                gold_pids.append(pid)

        all_sampled_queries.append({
            "query_id": qid,
            "query": qtext,
            "query_type": qtype,
            "answers": answers,
            "wellFormedAnswers": well_formed,
            "gold_pids": gold_pids,
            "candidate_pids": candidate_pids
        })

        if query_count % 1000 == 0:
            print(f"Processed {query_count} queries | unique passages: {len(unique_passages):,} | elapsed: {time.time() - t0:.1f}s")

        if len(unique_passages) >= 100000:
            break

    elapsed = time.time() - t0
    print(f"\nExtraction complete in {elapsed:.1f}s!")
    print(f"Total queries sampled: {query_count}")
    print(f"Total candidate passages examined: {total_candidates_seen:,}")
    print(f"Unique passages indexed: {len(unique_passages):,}")
    print(f"Duplicate passages pruned: {duplicate_candidates_count:,} ({duplicate_candidates_count/total_candidates_seen*100:.2f}%)")

    # Save passages jsonl
    passages_list = list(unique_passages.values())
    passages_file = out_dir / "c100k_raw_passages.jsonl"
    print(f"Saving {len(passages_list):,} passages to {passages_file}...")
    with open(passages_file, "w", encoding="utf-8") as f:
        for p in passages_list:
            f.write(json.dumps(p) + "\n")

    # Sample TUNE (500) and BENCH (100) disjoint
    import random
    random.seed(42)
    # Select only queries with at least one gold passage
    queries_with_gold = [q for q in all_sampled_queries if len(q["gold_pids"]) > 0]
    print(f"Queries with at least one gold passage: {len(queries_with_gold)} / {len(all_sampled_queries)}")

    random.shuffle(queries_with_gold)
    tune_queries = queries_with_gold[:500]
    bench_queries = queries_with_gold[500:600]

    with open(out_dir / "tune_raw_500.json", "w", encoding="utf-8") as f:
        json.dump(tune_queries, f, indent=2)

    with open(out_dir / "bench_raw_100.json", "w", encoding="utf-8") as f:
        json.dump(bench_queries, f, indent=2)

    # Compute statistics
    query_types = [p["category"] for p in passages_list]
    type_counts = Counter(query_types)
    
    total_positives = sum(len(q["gold_pids"]) for q in all_sampled_queries)
    pos_ratio = total_positives / total_candidates_seen

    stats = {
        "dataset_name": "c100k_raw",
        "source": "microsoft/ms_marco v2.1 validation",
        "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries_sampled": query_count,
        "total_candidates_seen": total_candidates_seen,
        "unique_passages_count": len(passages_list),
        "duplicates_pruned": duplicate_candidates_count,
        "duplicate_percentage": round(duplicate_candidates_count / total_candidates_seen * 100, 2),
        "positive_to_total_ratio": round(pos_ratio, 4),
        "total_positive_annotations": total_positives,
        "per_query_type_counts": dict(type_counts.most_common()),
        "tune_queries_count": len(tune_queries),
        "bench_queries_count": len(bench_queries),
        "extraction_duration_seconds": round(elapsed, 2)
    }

    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDataset Statistics Summary:")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    extract_c100k_raw()
