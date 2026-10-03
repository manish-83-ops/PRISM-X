"""Extract and audit MS MARCO human reference answers for 100 BENCH queries."""

from __future__ import annotations

import json
from pathlib import Path
import datasets

REPO_ROOT = Path(__file__).resolve().parent.parent

def is_valid_answer(ans: str | None) -> bool:
    if not ans:
        return False
    ans_clean = ans.strip().lower()
    if ans_clean in ("", "no answer present", "no answer present.", "none", "n/a", "null"):
        return False
    # Exclude single non-informative tokens (e.g. "?", ".", "-")
    if len(ans_clean.split()) < 2 and len(ans_clean) < 4:
        return False
    return True

def select_reference_answer(row: dict) -> tuple[str | None, str]:
    """Answer-selection rule:
    1. Use wellFormedAnswers[0] if present and valid.
    2. Otherwise use answers[0] if present and valid.
    3. Otherwise return None and exclusion reason.
    """
    wf_list = row.get("wellFormedAnswers") or []
    for wf in wf_list:
        if is_valid_answer(wf):
            return wf.strip(), "wellFormedAnswers"

    ans_list = row.get("answers") or []
    for a in ans_list:
        if is_valid_answer(a):
            return a.strip(), "answers"

    # Determine reason for exclusion
    if not wf_list and not ans_list:
        return None, "empty_answers_list"
    
    first_ans = (ans_list[0] if ans_list else "").strip()
    if first_ans.lower().startswith("no answer present"):
        return None, "no_answer_present"
    return None, f"invalid_content: {first_ans[:30]}"

def main():
    bench_file = REPO_ROOT / "data" / "manifests" / "split_bench.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    bench_map = {int(q["query_id"]): q for q in bench_queries}
    bench_ids = set(bench_map.keys())

    print(f"Loaded {len(bench_queries)} BENCH queries. Streaming MS MARCO v2.1 validation split...")
    ds = datasets.load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)

    extracted_rows = {}
    for row in ds:
        qid = row["query_id"]
        if qid in bench_ids:
            extracted_rows[qid] = row
            if len(extracted_rows) == len(bench_ids):
                break

    print(f"Matched {len(extracted_rows)} / {len(bench_ids)} queries in MS MARCO v2.1.\n")

    # Evaluate coverage in seeded order
    valid_queries = []
    excluded_queries = []
    annotated_bench = []

    for item in bench_queries:
        qid = int(item["query_id"])
        row = extracted_rows.get(qid, {})
        ref_ans, source_field = select_reference_answer(row)

        record = {
            "query_id": str(qid),
            "query": item["query"],
            "gold_passage_ids": item["gold_passage_ids"],
            "raw_answers": row.get("answers", []),
            "raw_well_formed_answers": row.get("wellFormedAnswers", []),
            "reference_answer": ref_ans,
            "answer_source": source_field,
            "is_valid": ref_ans is not None,
        }
        annotated_bench.append(record)

        if ref_ans is not None:
            valid_queries.append(record)
        else:
            excluded_queries.append(record)

    # Check existing N=25 run
    old_checkpoint_file = REPO_ROOT / "results" / "ragas" / "paired_checkpoint.json"
    old_qids = []
    if old_checkpoint_file.exists():
        with open(old_checkpoint_file, "r", encoding="utf-8") as f:
            old_ckpt = json.load(f)
            if "completed_queries" in old_ckpt:
                old_qids = [str(k) for k in old_ckpt["completed_queries"].keys()]
            elif isinstance(old_ckpt, list):
                old_qids = [str(e["query_id"]) for e in old_ckpt]

    old_in_valid = [qid for qid in old_qids if qid in set(q["query_id"] for q in valid_queries)]
    old_excluded = [qid for qid in old_qids if qid in set(q["query_id"] for q in excluded_queries)]

    # Summary
    print("====================================================================")
    print("MS MARCO REFERENCE ANSWER COVERAGE AUDIT (100 BENCH QUERIES)")
    print("====================================================================")
    print(f"Total BENCH Queries:            {len(bench_queries)}")
    print(f"Valid Reference Answers:        {len(valid_queries)} ({len(valid_queries)/len(bench_queries)*100:.1f}%)")
    print(f"Excluded Queries:               {len(excluded_queries)} ({len(excluded_queries)/len(bench_queries)*100:.1f}%)")
    
    exclusion_reasons = {}
    for eq in excluded_queries:
        r = eq["answer_source"]
        exclusion_reasons[r] = exclusion_reasons.get(r, 0) + 1
    print("\nExclusion Breakdown:")
    for r, count in exclusion_reasons.items():
        print(f"  - {r}: {count} queries")

    print(f"\nExisting N=25 Run Analysis:")
    print(f"  - Total queries in old run:      {len(old_qids)}")
    print(f"  - Inside valid-answer set:       {len(old_in_valid)} / {len(old_qids)}")
    print(f"  - Excluded from valid-answer set:{len(old_excluded)} / {len(old_qids)}")
    if old_excluded:
        print(f"    Excluded query IDs in old run: {old_excluded}")

    # Freeze the first 25 valid-answer queries in seeded order
    frozen_valid_25 = valid_queries[:25]
    print(f"\nFrozen N=25 Query Set (First 25 valid queries in seeded order):")
    for idx, q in enumerate(frozen_valid_25, start=1):
        print(f"  {idx:02d}. QID: {q['query_id']} | Source: {q['answer_source']} | Ans: {q['reference_answer'][:60]}...")

    # Save artifacts
    out_file = REPO_ROOT / "data" / "manifests" / "bench_reference_answers.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(annotated_bench, f, indent=2)

    frozen_file = REPO_ROOT / "data" / "manifests" / "frozen_ragas_bench_queries.json"
    with open(frozen_file, "w", encoding="utf-8") as f:
        json.dump({
            "audit_summary": {
                "total_bench": len(bench_queries),
                "valid_count": len(valid_queries),
                "excluded_count": len(excluded_queries),
                "exclusion_breakdown": exclusion_reasons,
                "old_25_valid_overlap": len(old_in_valid),
                "old_25_excluded_overlap": len(old_excluded),
            },
            "selection_rule": "Use wellFormedAnswers[0] if present and valid; else answers[0] if present and valid ('No Answer Present', empty, or single-token excluded).",
            "frozen_queries_n25": frozen_valid_25,
            "all_valid_queries": valid_queries,
        }, f, indent=2)

    print(f"\nSaved annotated reference answers to {out_file}")
    print(f"Saved frozen RAGAS query set to {frozen_file}")

if __name__ == "__main__":
    main()
