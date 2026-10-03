"""Resample c100k_raw with fixed seed 42 from microsoft/ms_marco v2.1 validation split.
In strict accordance with Gate 5.2 instructions (1a, 1b):
1. Seeded random permutation (seed=42) of all 101,093 validation queries.
2. Ingest candidate passages, deduplicate by normalized text, stop when unique passages >= 100,000.
3. Compare query_type distribution of sampled queries vs full 101,093 split (verify < 2% difference).
4. Check eligibility:
   - Queries with >= 1 is_selected gold passage
   - Queries with valid ADR-014 human reference answer
5. Draw TUNE (500, seed=42) and BENCH (100, seed=1337) strictly from eligible queries (disjoint).
6. Save updated artifacts and manifests.
"""

from collections import Counter
import json
from pathlib import Path
import random
import time
from datasets import load_dataset


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def check_adr014_answer(answers, well_formed) -> str | None:
    def check_ans(a_list):
        if not a_list:
            return None
        ans = a_list[0].strip()
        if not ans:
            return None
        if ans.lower() in ["no answer present", "no answer present.", "yes", "no"]:
            return None
        tokens = ans.split()
        if len(tokens) <= 1:
            return None
        return ans

    wf_ans = check_ans(well_formed)
    if wf_ans:
        return wf_ans
    return check_ans(answers)


def main():
    out_dir = Path("data/c100k_raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached validation dataset...")
    t0 = time.time()
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    n_total_queries = len(ds)
    print(f"Loaded {n_total_queries} queries in {time.time() - t0:.2f}s.")

    # 1. Full split distribution
    full_counts = Counter(ds["query_type"])
    full_dist = {k: round(v / n_total_queries * 100, 2) for k, v in full_counts.most_common()}
    print(f"\nFull Validation Split ({n_total_queries} queries) query_type distribution:")
    for k, pct in full_dist.items():
        print(f"  {k}: {full_counts[k]:,} ({pct:.2f}%)")

    # 2. Seeded Random Permutation (seed = 42)
    random.seed(42)
    shuffled_indices = list(range(n_total_queries))
    random.shuffle(shuffled_indices)

    unique_passages = {}
    sampled_queries_meta = []
    total_candidates_seen = 0
    duplicate_candidates = 0

    print("\nSampling queries sequentially along seed=42 permutation...")
    t_sample_start = time.time()

    for idx in shuffled_indices:
        item = ds[idx]
        qid = item["query_id"]
        qtext = item["query"]
        qtype = item["query_type"]
        answers = item.get("answers", [])
        well_formed = item.get("wellFormedAnswers", [])

        valid_ref = check_adr014_answer(answers, well_formed)
        p_dict = item["passages"]
        p_texts = p_dict["passage_text"]
        p_urls = p_dict["url"]
        p_is_sel = p_dict["is_selected"]

        gold_pids = []
        cand_pids = []

        for text, url, is_sel in zip(p_texts, p_urls, p_is_sel):
            total_candidates_seen += 1
            norm = normalize_text(text)
            if norm in unique_passages:
                duplicate_candidates += 1
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

            cand_pids.append(pid)
            if is_sel == 1:
                gold_pids.append(pid)

        has_gold = len(gold_pids) > 0
        is_eligible = has_gold and (valid_ref is not None)

        sampled_queries_meta.append({
            "query_id": qid,
            "query": qtext,
            "query_type": qtype,
            "answers": answers,
            "wellFormedAnswers": well_formed,
            "valid_reference_answer": valid_ref,
            "has_gold": has_gold,
            "is_eligible": is_eligible,
            "gold_pids": gold_pids,
            "candidate_pids": cand_pids
        })

        if len(unique_passages) >= 100000:
            break

    n_sampled = len(sampled_queries_meta)
    print(f"Sampling complete in {time.time() - t_sample_start:.2f}s!")
    print(f"Total queries sampled: {n_sampled:,}")
    print(f"Unique passages indexed: {len(unique_passages):,}")
    print(f"Total candidate passages examined: {total_candidates_seen:,}")
    print(f"Duplicates pruned: {duplicate_candidates:,} ({duplicate_candidates/total_candidates_seen*100:.2f}%)")

    # 3. Verify Query Type Distribution Match (Step 1a)
    sampled_counts = Counter(q["query_type"] for q in sampled_queries_meta)
    sampled_dist = {k: round(v / n_sampled * 100, 2) for k, v in sampled_counts.most_common()}

    print("\n=================================================================")
    print("STEP 1a: QUERY TYPE DISTRIBUTION COMPARISON")
    print(f"{'Query Type':<15} | {'Full Split %':<15} | {'Sampled (Seed 42) %':<20} | {'Abs Diff (%)':<15} | Status")
    print("-" * 80)
    all_within_2pct = True
    for k in full_counts:
        f_pct = full_dist.get(k, 0.0)
        s_pct = sampled_dist.get(k, 0.0)
        diff = abs(s_pct - f_pct)
        status = "PASS (<2%)" if diff <= 2.0 else "FAIL (>2%)"
        if diff > 2.0:
            all_within_2pct = False
        print(f"{k:<15} | {f_pct:>13.2f}% | {s_pct:>18.2f}% | {diff:>13.2f}% | {status}")
    print(f"Overall Distribution Check: {'PASS (All within 2 percentage points)' if all_within_2pct else 'FAIL'}")

    # 4. Eligibility Counts (Step 1b)
    n_with_gold = sum(1 for q in sampled_queries_meta if q["has_gold"])
    n_with_valid_ans = sum(1 for q in sampled_queries_meta if q["valid_reference_answer"] is not None)
    n_eligible = sum(1 for q in sampled_queries_meta if q["is_eligible"])

    print("\n=================================================================")
    print("STEP 1b: ELIGIBILITY AUDIT")
    print(f"Total sampled queries in c100k_raw: {n_sampled:,}")
    print(f"Queries with >= 1 is_selected passage: {n_with_gold:,} ({n_with_gold/n_sampled*100:.2f}%)")
    print(f"Queries with valid ADR-014 reference answer: {n_with_valid_ans:,} ({n_with_valid_ans/n_sampled*100:.2f}%)")
    print(f"ELIGIBLE queries (>= 1 gold AND valid answer): {n_eligible:,} ({n_eligible/n_sampled*100:.2f}%)")

    # 5. Draw TUNE (500) and BENCH (100) Disjoint
    eligible_queries = [q for q in sampled_queries_meta if q["is_eligible"]]
    assert len(eligible_queries) >= 600, f"Insufficient eligible queries ({len(eligible_queries)})"

    # Seed 42 for TUNE
    rng_tune = random.Random(42)
    shuffled_eligible = list(eligible_queries)
    rng_tune.shuffle(shuffled_eligible)
    tune_500 = shuffled_eligible[:500]

    # Remaining eligible for BENCH (seed 1337)
    remaining_eligible = shuffled_eligible[500:]
    rng_bench = random.Random(1337)
    rng_bench.shuffle(remaining_eligible)
    bench_100 = remaining_eligible[:100]

    tune_ids = {q["query_id"] for q in tune_500}
    bench_ids = {q["query_id"] for q in bench_100}
    assert len(tune_ids & bench_ids) == 0, "TUNE and BENCH must be strictly disjoint!"
    print(f"Successfully drawn TUNE (500) and BENCH (100) disjoint from {len(eligible_queries)} eligible pool.")

    # 6. Save Manifests and Files
    passages_list = list(unique_passages.values())
    passages_file = out_dir / "c100k_raw_passages.jsonl"
    print(f"\nWriting {len(passages_list):,} unique passages to {passages_file}...")
    with open(passages_file, "w", encoding="utf-8") as f:
        for p in passages_list:
            f.write(json.dumps(p) + "\n")

    with open(out_dir / "tune_raw_500.json", "w", encoding="utf-8") as f:
        json.dump(tune_500, f, indent=2)

    with open(out_dir / "bench_raw_100.json", "w", encoding="utf-8") as f:
        json.dump(bench_100, f, indent=2)

    total_positives = sum(len(q["gold_pids"]) for q in sampled_queries_meta)
    stats = {
        "dataset_name": "c100k_raw",
        "source": "microsoft/ms_marco v2.1 validation",
        "sampling_method": "seeded_random_permutation (seed=42)",
        "total_queries_sampled": n_sampled,
        "total_candidates_seen": total_candidates_seen,
        "unique_passages_count": len(passages_list),
        "duplicates_pruned": duplicate_candidates,
        "duplicate_percentage": round(duplicate_candidates / total_candidates_seen * 100, 2),
        "queries_with_gold_count": n_with_gold,
        "queries_with_valid_answer_count": n_with_valid_ans,
        "eligible_queries_count": n_eligible,
        "positive_to_total_ratio": round(total_positives / total_candidates_seen, 4),
        "per_query_type_counts": dict(sampled_counts.most_common()),
        "query_type_distribution_comparison": {
            k: {
                "full_split_pct": full_dist.get(k, 0.0),
                "sampled_pct": sampled_dist.get(k, 0.0),
                "abs_diff_pct": round(abs(sampled_dist.get(k, 0.0) - full_dist.get(k, 0.0)), 2)
            }
            for k in full_counts
        },
        "tune_queries_count": len(tune_500),
        "bench_queries_count": len(bench_100),
    }

    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDataset stats written to data/c100k_raw/dataset_stats.json.")


if __name__ == "__main__":
    main()
