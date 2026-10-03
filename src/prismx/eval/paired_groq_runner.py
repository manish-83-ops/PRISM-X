"""Paired Groq LLM-as-a-judge RAGAS evaluation runner with checkpointing and token telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "manifests"
RESULTS_DIR = REPO_ROOT / "results" / "ragas"
CHECKPOINT_FILE = RESULTS_DIR / "paired_checkpoint.json"
INTERIM_SUMMARY_FILE = RESULTS_DIR / "interim_summary.json"

# Groq Free Tier Limits for openai/gpt-oss-20b
GROQ_LIMITS = {
    "model": "openai/gpt-oss-20b",
    "rpm": 30,
    "rpd": 1000,
    "tpd": 500000,
    "tpm": 8000,
}


def load_bench_queries(n: int = 100) -> list[dict[str, Any]]:
    bench_file = SPLITS_DIR / "split_bench.json"
    if not bench_file.exists():
        raise FileNotFoundError(f"Missing {bench_file}")
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data if isinstance(data, list) else data.get("queries", [])

    # Hydrate gold reference text from SQLite text store
    from prismx.index.text_store import TextStore
    db_path = REPO_ROOT / "data" / "text_store.db"
    if db_path.exists():
        store = TextStore(db_path=db_path)
        all_gold_ids = [str(q["gold_passage_ids"][0]) for q in queries if q.get("gold_passage_ids")]
        passages = store.get_passages_by_ids(all_gold_ids)
        for q in queries:
            gold_ids = q.get("gold_passage_ids", [])
            if gold_ids and str(gold_ids[0]) in passages:
                q["reference_text"] = passages[str(gold_ids[0])]["text"]
        store.close()

    return queries[:n]


def load_retrievals() -> tuple[dict[str, Any], dict[str, Any]]:
    p1_file = REPO_ROOT / "results" / "phase1" / "bench_dense_retrievals.json"
    p2_file = REPO_ROOT / "results" / "phase2" / "bench_hybrid_retrievals.json"
    if not p1_file.exists() or not p2_file.exists():
        raise FileNotFoundError("Retrieval result files missing in results/phase1 or results/phase2")
    with open(p1_file, "r", encoding="utf-8") as f:
        p1 = json.load(f)
    with open(p2_file, "r", encoding="utf-8") as f:
        p2 = json.load(f)

    if isinstance(p1, list):
        p1 = {str(item["query_id"]): item for item in p1}
    if isinstance(p2, list):
        p2 = {str(item["query_id"]): item for item in p2}
    return p1, p2


class GroqTelemetryJudge:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set. Please set it before running.")
        from groq import Groq
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def call_llm(self, prompt: str) -> tuple[str, int, int]:
        max_retries = 15
        base_delay = 2.5

        for attempt in range(max_retries):
            try:
                # Respect 30 RPM (min 2.2s interval between sequential requests)
                time.sleep(2.2)
                chat = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=0.0,
                )
                output = chat.choices[0].message.content.strip()
                usage = getattr(chat, "usage", None)
                p_tok = usage.prompt_tokens if usage else int(len(prompt.split()) * 1.3)
                c_tok = usage.completion_tokens if usage else int(len(output.split()) * 1.3)
                self.total_prompt_tokens += p_tok
                self.total_completion_tokens += c_tok
                self.total_calls += 1
                return output, p_tok, c_tok
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str:
                    sleep_time = 5.0
                    m_min = re.search(r"try again in (\d+)m([\d\.]+)s", err_str)
                    m_sec = re.search(r"try again in ([\d\.]+)s", err_str)
                    if m_min:
                        mins = float(m_min.group(1))
                        secs = float(m_min.group(2))
                        sleep_time = mins * 60.0 + secs + 2.0
                    elif m_sec:
                        sleep_time = float(m_sec.group(1)) + 2.0
                    else:
                        sleep_time = base_delay * (2 ** min(attempt, 6)) + random.uniform(1.0, 3.0)

                    print(f"  [429 Rate Limit] Backing off for {sleep_time:.2f}s (attempt {attempt+1}/{max_retries})...", flush=True)
                    time.sleep(sleep_time)
                else:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(3.0)

        raise RuntimeError("Max retries exceeded on Groq API.")

    def evaluate_context_precision(self, question: str, retrieved_contexts: list[str], reference: str) -> tuple[float, int, int]:
        """Evaluates RAGAS Context Precision: rank-weighted precision over the top-5 retrieved passages."""
        top5 = retrieved_contexts[:5]
        if not top5:
            return 0.0, 0, 0

        passages_text = "\n".join([f"Passage [{i+1}]: {ctx}" for i, ctx in enumerate(top5)])
        prompt = (
            f"You are an impartial evaluator for information retrieval systems.\n"
            f"Question: {question}\n"
            f"Reference Ground Truth: {reference}\n\n"
            f"Retrieved Passages:\n{passages_text}\n\n"
            f"For each passage [1] to [{len(top5)}], determine if it contains information directly useful to answer the Question.\n"
            f"Respond with ONLY a JSON list of {len(top5)} booleans, e.g. [true, false, true, false, false]. Nothing else.\n"
            f"JSON:"
        )
        resp, p_tok, c_tok = self.call_llm(prompt)

        # Parse boolean judgments
        match = re.search(r"\[.*?\]", resp, re.DOTALL)
        verdicts = []
        if match:
            try:
                verdicts = json.loads(match.group(0))
            except Exception:
                pass
        if not verdicts or len(verdicts) != len(top5):
            verdicts = [("true" in w.lower() or "yes" in w.lower()) for w in resp.split()[:len(top5)]]
            if len(verdicts) < len(top5):
                verdicts += [False] * (len(top5) - len(verdicts))

        # Compute standard RAGAS Context Precision formula
        hits = 0
        cum_prec = 0.0
        for rank, is_rel in enumerate(verdicts[:len(top5)], start=1):
            if is_rel:
                hits += 1
                cum_prec += hits / rank

        cp = (cum_prec / hits) if hits > 0 else 0.0
        return round(cp, 4), p_tok, c_tok

    def evaluate_context_recall(self, question: str, retrieved_contexts: list[str], reference: str) -> tuple[float, int, int]:
        """Evaluates RAGAS Context Recall: fraction of reference assertions answered by retrieved passages."""
        top5 = retrieved_contexts[:5]
        joined = "\n---\n".join(top5)
        prompt = (
            f"You are an impartial evaluator for information retrieval systems.\n"
            f"Question: {question}\n"
            f"Reference Ground Truth: {reference}\n\n"
            f"Retrieved Passages:\n{joined}\n\n"
            f"Can the key factual assertions in the Reference Ground Truth be answered using the Retrieved Passages?\n"
            f"Reply with a single float score between 0.0 (none covered) and 1.0 (fully covered), followed by nothing else.\n"
            f"Score:"
        )
        resp, p_tok, c_tok = self.call_llm(prompt)
        try:
            val = float(resp.split()[0].strip())
            score = max(0.0, min(1.0, val))
        except Exception:
            score = 1.0 if ("1" in resp or "YES" in resp.upper()) else 0.0
        return round(score, 4), p_tok, c_tok


def run_smoke_test(judge: GroqTelemetryJudge, bench_queries: list[dict[str, Any]], p1_ret: dict, p2_ret: dict):
    print("=" * 70)
    print("GROQ RAGAS 3-QUERY SMOKE TEST")
    print("=" * 70)
    smoke_queries = bench_queries[:3]
    total_tokens = 0
    total_calls = 0
    t0 = time.time()

    for idx, q in enumerate(smoke_queries, start=1):
        qid = str(q["query_id"])
        query_text = q["query"]
        ref_text = q.get("reference_text", "")

        # Phase 1
        p1_item = p1_ret.get(qid, {})
        p1_contexts = p1_item.get("retrieved_texts", [])
        _, pt1, ct1 = judge.evaluate_context_precision(query_text, p1_contexts, ref_text)
        _, pt2, ct2 = judge.evaluate_context_recall(query_text, p1_contexts, ref_text)

        # Phase 2
        p2_item = p2_ret.get(qid, {})
        p2_contexts = p2_item.get("retrieved_texts", [])
        _, pt3, ct3 = judge.evaluate_context_precision(query_text, p2_contexts, ref_text)
        _, pt4, ct4 = judge.evaluate_context_recall(query_text, p2_contexts, ref_text)

        q_tokens = pt1 + ct1 + pt2 + ct2 + pt3 + ct3 + pt4 + ct4
        total_tokens += q_tokens
        total_calls += 4
        print(f"Query {idx}/3 (qid={qid}): 4 LLM calls, {q_tokens} tokens consumed")

    elapsed = time.time() - t0
    avg_tokens_per_query = total_tokens / len(smoke_queries)
    est_100_queries_tokens = avg_tokens_per_query * 100
    daily_capacity_queries = int(GROQ_LIMITS["tpd"] / avg_tokens_per_query)
    days_for_100 = est_100_queries_tokens / GROQ_LIMITS["tpd"]

    smoke_summary = {
        "model": judge.model_name,
        "n_smoke_queries": len(smoke_queries),
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "avg_tokens_per_query": round(avg_tokens_per_query, 1),
        "avg_calls_per_query": 4,
        "elapsed_seconds": round(elapsed, 1),
        "free_tier_limits": GROQ_LIMITS,
        "projections": {
            "est_100_queries_tokens": int(est_100_queries_tokens),
            "daily_capacity_queries": daily_capacity_queries,
            "days_for_100_queries": round(days_for_100, 2),
            "fits_in_one_day": days_for_100 <= 1.0,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    smoke_file = RESULTS_DIR / "smoke_test_summary.json"
    with open(smoke_file, "w", encoding="utf-8") as f:
        json.dump(smoke_summary, f, indent=2)

    print("\nSmoke Test Results:")
    print(f"  Avg Tokens per Paired Query: {avg_tokens_per_query:.1f}")
    print(f"  Est. Tokens for 100 Queries: {est_100_queries_tokens:,.0f} (Daily limit: {GROQ_LIMITS['tpd']:,})")
    print(f"  Queries Fitting in Daily Limit: {daily_capacity_queries}")
    print(f"  Est. Days for Full 100 Run: {days_for_100:.2f} days")
    print(f"  Fits within 1 day: {days_for_100 <= 1.0}")
    print(f"  Smoke summary written to: {smoke_file}")
    print("=" * 70)


def run_paired_evaluation(
    judge: GroqTelemetryJudge,
    bench_queries: list[dict[str, Any]],
    p1_ret: dict,
    p2_ret: dict,
    chunk_size: int = 25,
    max_queries: int = 100,
):
    from prismx.eval.bootstrap import bootstrap_ci, paired_bootstrap_difference

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {"completed_queries": {}}
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"Resuming from checkpoint: {len(checkpoint.get('completed_queries', {}))} queries already completed.")
        except Exception as e:
            print(f"Could not read checkpoint, starting fresh: {e}")

    completed = checkpoint.setdefault("completed_queries", {})
    queries_to_run = bench_queries[:max_queries]
    print(f"\nStarting Paired Groq RAGAS Evaluation on {len(queries_to_run)} queries (Chunk size: {chunk_size})...")

    chunk_count = len(completed) // chunk_size

    for idx, q in enumerate(queries_to_run, start=1):
        qid = str(q["query_id"])
        if qid in completed:
            continue

        query_text = q["query"]
        ref_text = q.get("reference_text", "")

        # Phase 1
        p1_item = p1_ret.get(qid, {})
        p1_contexts = p1_item.get("retrieved_texts", [])
        p1_cp, _, _ = judge.evaluate_context_precision(query_text, p1_contexts, ref_text)
        p1_cr, _, _ = judge.evaluate_context_recall(query_text, p1_contexts, ref_text)

        # Phase 2
        p2_item = p2_ret.get(qid, {})
        p2_contexts = p2_item.get("retrieved_texts", [])
        p2_cp, _, _ = judge.evaluate_context_precision(query_text, p2_contexts, ref_text)
        p2_cr, _, _ = judge.evaluate_context_recall(query_text, p2_contexts, ref_text)

        # Checkpoint immediately after each query
        completed[qid] = {
            "query_id": qid,
            "query": query_text,
            "phase1": {"context_precision": p1_cp, "context_recall": p1_cr},
            "phase2": {"context_precision": p2_cp, "context_recall": p2_cr},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  [{len(completed)}/{len(queries_to_run)}] qid={qid} | P1: CP={p1_cp:.2f}, CR={p1_cr:.2f} | P2: CP={p2_cp:.2f}, CR={p2_cr:.2f}", flush=True)

        if len(completed) % chunk_size == 0:
            chunk_count = len(completed) // chunk_size
            write_interim_summary(completed, chunk_count, len(queries_to_run))

    # Final summary
    write_interim_summary(completed, chunk_count + 1, len(queries_to_run), is_final=True)


def write_interim_summary(completed: dict, chunk_idx: int, total_queries: int, is_final: bool = False):
    from prismx.eval.bootstrap import bootstrap_ci, paired_bootstrap_difference

    p1_cps = [v["phase1"]["context_precision"] for v in completed.values()]
    p1_crs = [v["phase1"]["context_recall"] for v in completed.values()]
    p2_cps = [v["phase2"]["context_precision"] for v in completed.values()]
    p2_crs = [v["phase2"]["context_recall"] for v in completed.values()]

    p1_cp_ci = bootstrap_ci(p1_cps, n_resamples=10000, seed=42)
    p1_cr_ci = bootstrap_ci(p1_crs, n_resamples=10000, seed=42)
    p2_cp_ci = bootstrap_ci(p2_cps, n_resamples=10000, seed=42)
    p2_cr_ci = bootstrap_ci(p2_crs, n_resamples=10000, seed=42)

    diff_cp = paired_bootstrap_difference(p1_cps, p2_cps, n_resamples=10000, seed=42)
    diff_cr = paired_bootstrap_difference(p1_crs, p2_crs, n_resamples=10000, seed=42)

    summary = {
        "chunk_index": chunk_idx,
        "is_final": is_final,
        "queries_completed": len(completed),
        "total_target_queries": total_queries,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase1": {
            "context_precision": p1_cp_ci,
            "context_recall": p1_cr_ci,
        },
        "phase2": {
            "context_precision": p2_cp_ci,
            "context_recall": p2_cr_ci,
        },
        "paired_difference_phase2_minus_phase1": {
            "context_precision": diff_cp,
            "context_recall": diff_cr,
        },
        "acceptance_status": {
            "nfr1_cp_target_gt_0_75": p2_cp_ci["mean"] > 0.75,
            "nfr2_cr_target_gt_0_70": p2_cr_ci["mean"] > 0.70,
        }
    }

    out_file = RESULTS_DIR / f"summary_{'final' if is_final else f'chunk_{chunk_idx}'}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(INTERIM_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n>>> Interim Summary (Chunk {chunk_idx}, N={len(completed)}/{total_queries}):")
    print(f"    Phase 1 Dense:  Context Precision = {p1_cp_ci['mean']:.4f} [{p1_cp_ci['ci_lower']:.4f}, {p1_cp_ci['ci_upper']:.4f}], Context Recall = {p1_cr_ci['mean']:.4f} [{p1_cr_ci['ci_lower']:.4f}, {p1_cr_ci['ci_upper']:.4f}]")
    print(f"    Phase 2 Hybrid: Context Precision = {p2_cp_ci['mean']:.4f} [{p2_cp_ci['ci_lower']:.4f}, {p2_cp_ci['ci_upper']:.4f}], Context Recall = {p2_cr_ci['mean']:.4f} [{p2_cr_ci['ci_lower']:.4f}, {p2_cr_ci['ci_upper']:.4f}]")
    print(f"    CP Delta (P2 - P1): {diff_cp['mean_diff']:+.4f} [{diff_cp['ci_lower']:+.4f}, {diff_cp['ci_upper']:+.4f}] (Significant: {diff_cp['is_statistically_distinguishable']})")
    print(f"    CR Delta (P2 - P1): {diff_cr['mean_diff']:+.4f} [{diff_cr['ci_lower']:+.4f}, {diff_cr['ci_upper']:+.4f}] (Significant: {diff_cr['is_statistically_distinguishable']})\n")


def main():
    parser = argparse.ArgumentParser(description="PRISMX Paired Groq RAGAS Evaluator")
    parser.add_argument("--smoke-test", action="store_true", help="Run 3-query smoke test to measure real token/call cost")
    parser.add_argument("--run", action="store_true", help="Run paired evaluation across BENCH queries with checkpointing")
    parser.add_argument("--chunk-size", type=int, default=25, help="Chunk size for interim summaries")
    parser.add_argument("--max-queries", type=int, default=100, help="Maximum number of queries to evaluate")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Groq model name")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    judge = GroqTelemetryJudge(model_name=args.model)
    bench_queries = load_bench_queries(args.max_queries)
    p1_ret, p2_ret = load_retrievals()

    if args.smoke_test:
        run_smoke_test(judge, bench_queries, p1_ret, p2_ret)
    elif args.run:
        run_paired_evaluation(
            judge,
            bench_queries,
            p1_ret,
            p2_ret,
            chunk_size=args.chunk_size,
            max_queries=args.max_queries,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
