"""Paired Groq LLM-as-a-judge RAGAS evaluation runner with checkpointing and token telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results" / "ragas"
CHECKPOINT_FILE = RESULTS_DIR / "paired_checkpoint.json"
INTERIM_SUMMARY_FILE = RESULTS_DIR / "interim_summary.json"

# Groq Free Tier Limits for llama-3.1-8b-instant
GROQ_LIMITS = {
    "model": "llama-3.1-8b-instant",
    "rpm": 30,
    "rpd": 14400,
    "tpd": 500000,
    "tpm": 6000,
}


def load_bench_queries(n: int = 100) -> list[dict[str, Any]]:
    bench_file = SPLITS_DIR / "split_bench.json"
    if not bench_file.exists():
        raise FileNotFoundError(f"Missing {bench_file}")
    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data if isinstance(data, list) else data.get("queries", [])
    # Return fixed seeded order
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
    return p1, p2


class GroqTelemetryJudge:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
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
        max_retries = 6
        base_delay = 2.5

        for attempt in range(max_retries):
            try:
                # Obey 30 RPM (min 2.0s interval between sequential calls)
                time.sleep(2.1)
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
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                    print(f"  [429 Rate Limit] Backing off for {sleep_time:.2f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(3.0)

        raise RuntimeError("Max retries exceeded on Groq API.")

    def evaluate_precision(self, question: str, top1_context: str, reference: str) -> tuple[float, int, int]:
        prompt = (
            f"You are an impartial evaluator for information retrieval systems.\n"
            f"Question: {question}\n"
            f"Reference Ground Truth: {reference}\n"
            f"Retrieved Top Passage: {top1_context}\n\n"
            f"Does the Retrieved Top Passage contain information directly useful for answering the Question?\n"
            f"Reply with exactly one word: 'YES' or 'NO'."
        )
        resp, p_tok, c_tok = self.call_llm(prompt)
        score = 1.0 if "YES" in resp.upper() else 0.0
        return score, p_tok, c_tok

    def evaluate_recall(self, question: str, retrieved_contexts: list[str], reference: str) -> tuple[float, int, int]:
        joined = "\n---\n".join(retrieved_contexts)
        prompt = (
            f"You are an impartial evaluator for information retrieval systems.\n"
            f"Question: {question}\n"
            f"Reference Ground Truth: {reference}\n"
            f"Retrieved Passages:\n{joined}\n\n"
            f"Can the key factual assertions in the Reference Ground Truth be answered using the Retrieved Passages?\n"
            f"Reply with a single float score between 0.0 (none) and 1.0 (fully covered), followed by nothing else.\n"
            f"Score:"
        )
        resp, p_tok, c_tok = self.call_llm(prompt)
        try:
            val = float(resp.split()[0].strip())
            score = max(0.0, min(1.0, val))
        except Exception:
            score = 1.0 if ("1" in resp or "YES" in resp.upper()) else 0.0
        return score, p_tok, c_tok


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
        ref_text = q.get("reference_text") or (q.get("gold_passages", [""])[0] if q.get("gold_passages") else "")

        # P1
        p1_item = p1_ret.get(qid, {})
        p1_top1 = p1_item.get("retrieved_texts", [""])[0] if p1_item.get("retrieved_texts") else ""
        p1_all = p1_item.get("retrieved_texts", [])
        _, pt1, ct1 = judge.evaluate_precision(query_text, p1_top1, ref_text)
        _, pt2, ct2 = judge.evaluate_recall(query_text, p1_all, ref_text)

        # P2
        p2_item = p2_ret.get(qid, {})
        p2_top1 = p2_item.get("retrieved_texts", [""])[0] if p2_item.get("retrieved_texts") else ""
        p2_all = p2_item.get("retrieved_texts", [])
        _, pt3, ct3 = judge.evaluate_precision(query_text, p2_top1, ref_text)
        _, pt4, ct4 = judge.evaluate_recall(query_text, p2_all, ref_text)

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
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {"completed_queries": {}, "stats": {}}
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"Resuming from checkpoint: {len(checkpoint.get('completed_queries', {}))} queries already completed.")
        except Exception as e:
            print(f"Could not read checkpoint, starting fresh: {e}")

    completed = checkpoint.setdefault("completed_queries", {})

    print(f"\nStarting Paired Groq RAGAS Evaluation on {len(bench_queries)} queries (Chunk size: {chunk_size})...")

    chunk_count = 0
    for idx, q in enumerate(bench_queries, start=1):
        qid = str(q["query_id"])
        if qid in completed:
            continue

        query_text = q["query"]
        ref_text = q.get("reference_text") or (q.get("gold_passages", [""])[0] if q.get("gold_passages") else "")

        # P1
        p1_item = p1_ret.get(qid, {})
        p1_top1 = p1_item.get("retrieved_texts", [""])[0] if p1_item.get("retrieved_texts") else ""
        p1_all = p1_item.get("retrieved_texts", [])
        p1_cp, _, _ = judge.evaluate_precision(query_text, p1_top1, ref_text)
        p1_cr, _, _ = judge.evaluate_recall(query_text, p1_all, ref_text)

        # P2
        p2_item = p2_ret.get(qid, {})
        p2_top1 = p2_item.get("retrieved_texts", [""])[0] if p2_item.get("retrieved_texts") else ""
        p2_all = p2_item.get("retrieved_texts", [])
        p2_cp, _, _ = judge.evaluate_precision(query_text, p2_top1, ref_text)
        p2_cr, _, _ = judge.evaluate_recall(query_text, p2_all, ref_text)

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

        print(f"  [{len(completed)}/{len(bench_queries)}] qid={qid} | P1: CP={p1_cp:.2f}, CR={p1_cr:.2f} | P2: CP={p2_cp:.2f}, CR={p2_cr:.2f}")

        if len(completed) % chunk_size == 0:
            chunk_count += 1
            write_interim_summary(completed, chunk_count, len(bench_queries))

    # Final summary
    write_interim_summary(completed, chunk_count + 1, len(bench_queries), is_final=True)


def write_interim_summary(completed: dict, chunk_idx: int, total_queries: int, is_final: bool = False):
    p1_cps = [v["phase1"]["context_precision"] for v in completed.values()]
    p1_crs = [v["phase1"]["context_recall"] for v in completed.values()]
    p2_cps = [v["phase2"]["context_precision"] for v in completed.values()]
    p2_crs = [v["phase2"]["context_recall"] for v in completed.values()]

    import numpy as np
    summary = {
        "chunk_index": chunk_idx,
        "is_final": is_final,
        "queries_completed": len(completed),
        "total_target_queries": total_queries,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase1": {
            "context_precision_mean": round(float(np.mean(p1_cps)), 4) if p1_cps else 0.0,
            "context_recall_mean": round(float(np.mean(p1_crs)), 4) if p1_crs else 0.0,
        },
        "phase2": {
            "context_precision_mean": round(float(np.mean(p2_cps)), 4) if p2_cps else 0.0,
            "context_recall_mean": round(float(np.mean(p2_crs)), 4) if p2_crs else 0.0,
        },
    }

    out_file = RESULTS_DIR / f"summary_{'final' if is_final else f'chunk_{chunk_idx}'}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(INTERIM_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n>>> Interim Summary (Chunk {chunk_idx}, N={len(completed)}/{total_queries}):")
    print(f"    Phase 1 Dense:  Context Precision = {summary['phase1']['context_precision_mean']:.4f}, Context Recall = {summary['phase1']['context_recall_mean']:.4f}")
    print(f"    Phase 2 Hybrid: Context Precision = {summary['phase2']['context_precision_mean']:.4f}, Context Recall = {summary['phase2']['context_recall_mean']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="PRISMX Paired Groq RAGAS Evaluator")
    parser.add_argument("--smoke-test", action="store_true", help="Run 3-query smoke test to measure real token/call cost")
    parser.add_argument("--run", action="store_true", help="Run paired evaluation across BENCH queries with checkpointing")
    parser.add_argument("--chunk-size", type=int, default=25, help="Chunk size for interim summaries")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant", help="Groq model name")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        print("Please set your Groq free-tier API key in your shell:")
        print("  $env:GROQ_API_KEY=\"gsk_...\" (PowerShell)")
        print("  export GROQ_API_KEY=\"gsk_...\" (Linux/Mac)")
        sys.exit(1)

    judge = GroqTelemetryJudge(model_name=args.model)
    bench_queries = load_bench_queries(100)
    p1_ret, p2_ret = load_retrievals()

    if args.smoke_test:
        run_smoke_test(judge, bench_queries, p1_ret, p2_ret)
    elif args.run:
        run_paired_evaluation(judge, bench_queries, p1_ret, p2_ret, chunk_size=args.chunk_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
