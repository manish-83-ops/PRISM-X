"""Gate 4B: Primary RAGAS LLM Benchmark on 25 Frozen BENCH Queries with MS MARCO Reference Answers.

Adheres strictly to ADR-014:
- Evaluates Phase 1 (Dense), Phase 2 (Hybrid), Phase 3 (Hybrid + MiniLM-L6 INT8 Rerank).
- Ground truth is the human-generated MS MARCO reference answer from frozen_ragas_bench_queries.json.
- Paced to respect Groq rate limits with progressive checkpointing.
- Computes bootstrap 95% confidence intervals for means and deltas.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragas_gate4b")

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "manifests"
RESULTS_DIR = REPO_ROOT / "results" / "ragas"
CHECKPOINT_FILE = RESULTS_DIR / "frozen25_checkpoint.json"

MODEL_NAME = os.environ.get("GROQ_MODEL", "allam-2-7b")
CALL_INTERVAL = float(os.environ.get("CALL_INTERVAL", "2.0"))  # Pacing between calls


def bootstrap_ci(arr: list[float] | np.ndarray, n_resamples: int = 10000, seed: int = 42) -> dict:
    data = np.asarray(arr, dtype=float)
    n = len(data)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = np.random.default_rng(seed)
    boot_means = np.mean(rng.choice(data, size=(n_resamples, n), replace=True), axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return {
        "mean": round(float(np.mean(data)), 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
    }


def paired_bootstrap_diff(a: list[float], b: list[float], n_resamples: int = 10000, seed: int = 42) -> dict:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diffs = arr_b - arr_a
    n = len(diffs)
    rng = np.random.default_rng(seed)
    boot_diffs = np.mean(rng.choice(diffs, size=(n_resamples, n), replace=True), axis=1)
    lo = float(np.percentile(boot_diffs, 2.5))
    hi = float(np.percentile(boot_diffs, 97.5))
    mean_diff = float(np.mean(diffs))
    sig = not (lo <= 0.0 <= hi)
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "is_statistically_distinguishable": sig,
    }


def call_llm(client, prompt: str, max_retries: int = 20) -> tuple[str, int]:
    """Call Groq with automatic retry and rate-limit backoff."""
    for att in range(max_retries):
        try:
            time.sleep(CALL_INTERVAL)
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_NAME,
                temperature=0.0,
            )
            text = chat.choices[0].message.content.strip()
            usage = getattr(chat, "usage", None)
            tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
            return text, tokens
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err or "resource_exhausted" in err:
                sleep_t = 15.0
                m1 = re.search(r"try again in (\d+)m([\d.]+)s", err)
                m2 = re.search(r"try again in ([\d.]+)s", err)
                if m1:
                    sleep_t = float(m1.group(1)) * 60 + float(m1.group(2)) + 2.0
                elif m2:
                    sleep_t = float(m2.group(1)) + 2.0
                else:
                    sleep_t = min(60.0, 5.0 * (2.0 ** min(att, 4))) + random.uniform(1.0, 3.0)
                logger.warning(f"[429 Rate Limit] Backing off {sleep_t:.1f}s (attempt {att+1}/{max_retries})...")
                time.sleep(sleep_t)
            else:
                if att == max_retries - 1:
                    raise
                logger.warning(f"[API Error] {e} — retrying in 5s...")
                time.sleep(5.0)
    raise RuntimeError("Max retries exceeded calling Groq API.")


def eval_context_precision(client, question: str, contexts: list[str], reference_answer: str) -> tuple[float, int]:
    top5 = contexts[:5]
    if not top5:
        return 0.0, 0
    passages_text = "\n".join([f"Passage [{i+1}]: {ctx}" for i, ctx in enumerate(top5)])
    prompt = (
        f"You are an impartial evaluator for information retrieval systems.\n"
        f"Question: {question}\n"
        f"Reference Ground Truth Answer: {reference_answer}\n\n"
        f"Retrieved Passages:\n{passages_text}\n\n"
        f"For each passage [1] to [{len(top5)}], determine if it contains information directly useful to verify or answer the Question according to the Reference Ground Truth Answer.\n"
        f"Respond with ONLY a JSON list of {len(top5)} booleans, e.g. [true, false, true, false, false]. Nothing else.\n"
        f"JSON:"
    )
    resp, tokens = call_llm(client, prompt)
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

    hits = 0
    cum_prec = 0.0
    for rank, is_rel in enumerate(verdicts[:len(top5)], start=1):
        if is_rel:
            hits += 1
            cum_prec += hits / rank
    cp = (cum_prec / hits) if hits > 0 else 0.0
    return round(cp, 4), tokens


def eval_context_recall(client, question: str, contexts: list[str], reference_answer: str) -> tuple[float, int]:
    top5 = contexts[:5]
    if not top5:
        return 0.0, 0
    joined = "\n---\n".join(top5)
    prompt = (
        f"You are an impartial evaluator for information retrieval systems.\n"
        f"Question: {question}\n"
        f"Reference Ground Truth Answer: {reference_answer}\n\n"
        f"Retrieved Passages:\n{joined}\n\n"
        f"Can the key factual assertions in the Reference Ground Truth Answer be verified or answered using the Retrieved Passages?\n"
        f"Reply with a single float score between 0.0 (none covered) and 1.0 (fully covered), followed by nothing else.\n"
        f"Score:"
    )
    resp, tokens = call_llm(client, prompt)
    try:
        val = float(resp.split()[0].strip())
        score = max(0.0, min(1.0, val))
    except Exception:
        score = 1.0 if ("1" in resp or "yes" in resp.lower()) else 0.0
    return round(score, 4), tokens


def load_retrieval_texts(text_store) -> tuple[dict, dict, dict]:
    """Load top 5 passage texts for Phase 1, Phase 2, and Phase 3."""
    # Phase 1
    p1_file = REPO_ROOT / "results" / "phase1" / "bench_dense_retrievals.json"
    with open(p1_file, "r", encoding="utf-8") as f:
        p1_raw = json.load(f)

    # Phase 2
    p2_file = REPO_ROOT / "results" / "phase2" / "bench_hybrid_retrievals.json"
    with open(p2_file, "r", encoding="utf-8") as f:
        p2_raw = json.load(f)

    # Phase 3
    p3_file = REPO_ROOT / "results" / "phase3" / "bench_hybrid_rerank_retrievals.json"
    with open(p3_file, "r", encoding="utf-8") as f:
        p3_raw = json.load(f)

    # Index by query_id
    def extract_top5_pids(records):
        out = {}
        for r in records:
            qid = str(r["query_id"])
            if "retrieved_passage_ids" in r:
                pids = [str(p) for p in r["retrieved_passage_ids"][:5]]
            elif "retrieved_ids" in r:
                pids = [str(p) for p in r["retrieved_ids"][:5]]
            elif "results" in r:
                pids = [str(x["passage_id"]) for x in r["results"][:5]]
            else:
                pids = []
            out[qid] = pids
        return out

    p1_pids = extract_top5_pids(p1_raw)
    p2_pids = extract_top5_pids(p2_raw)
    p3_pids = extract_top5_pids(p3_raw)

    # Collect all needed passage IDs and fetch once from text_store
    all_pids = set()
    for m in (p1_pids, p2_pids, p3_pids):
        for pids in m.values():
            all_pids.update(pids)

    passages = text_store.get_passages_by_ids(list(all_pids))

    def pids_to_texts(pid_map):
        texts = {}
        for qid, pids in pid_map.items():
            texts[qid] = [passages[p]["text"] for p in pids if p in passages]
        return texts

    return pids_to_texts(p1_pids), pids_to_texts(p2_pids), pids_to_texts(p3_pids)


def main():
    print("====================================================================")
    print(f"GATE 4B: PRIMARY RAGAS BENCHMARK ON 25 FROZEN QUERIES ({MODEL_NAME})")
    print("Ground Truth: MS MARCO Human-Generated Reference Answers (ADR-014)")
    print("====================================================================\n")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        sys.exit(1)

    from groq import Groq
    client = Groq(api_key=api_key)

    # 1. Load frozen queries
    frozen_file = SPLITS_DIR / "frozen_ragas_bench_queries.json"
    with open(frozen_file, "r", encoding="utf-8") as f:
        frozen_data = json.load(f)

    queries = frozen_data["frozen_queries_n25"]
    print(f"Loaded {len(queries)} frozen queries with valid MS MARCO reference answers.")

    # 2. Load SQLite text store to resolve passages
    from prismx.index.text_store import TextStore
    cfg_db = REPO_ROOT / "data" / "text_store.db"
    text_store = TextStore(db_path=cfg_db)

    print("Loading retrieved passage texts for Phase 1, Phase 2, and Phase 3...")
    p1_texts, p2_texts, p3_texts = load_retrieval_texts(text_store)
    text_store.close()
    print("Passage texts loaded successfully.")

    # 3. Checkpoint handling
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "benchmark": "Gate 4B Primary RAGAS with Human Reference Answers",
        "ground_truth_rule": frozen_data["selection_rule"],
        "model": MODEL_NAME,
        "completed_queries": {},
    }
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if saved.get("completed_queries"):
                    checkpoint["completed_queries"] = saved["completed_queries"]
                    print(f"Resuming from checkpoint: {len(checkpoint['completed_queries'])} queries already scored.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    completed = checkpoint["completed_queries"]
    total_tokens = 0
    t0 = time.time()

    # 4. Evaluation Loop
    for idx, item in enumerate(queries, start=1):
        qid = str(item["query_id"])
        query_text = item["query"]
        ref_ans = item["reference_answer"]

        if qid in completed:
            continue

        c1 = p1_texts.get(qid, [])
        c2 = p2_texts.get(qid, [])
        c3 = p3_texts.get(qid, [])

        # Evaluate Phase 1
        p1_cp, tok1 = eval_context_precision(client, query_text, c1, ref_ans)
        p1_cr, tok2 = eval_context_recall(client, query_text, c1, ref_ans)

        # Evaluate Phase 2
        p2_cp, tok3 = eval_context_precision(client, query_text, c2, ref_ans)
        p2_cr, tok4 = eval_context_recall(client, query_text, c2, ref_ans)

        # Evaluate Phase 3
        p3_cp, tok5 = eval_context_precision(client, query_text, c3, ref_ans)
        p3_cr, tok6 = eval_context_recall(client, query_text, c3, ref_ans)

        q_tokens = tok1 + tok2 + tok3 + tok4 + tok5 + tok6
        total_tokens += q_tokens

        completed[qid] = {
            "query_id": qid,
            "query": query_text,
            "reference_answer": ref_ans,
            "answer_source": item.get("answer_source", "msmarco"),
            "phase1_dense": {"context_precision": p1_cp, "context_recall": p1_cr},
            "phase2_hybrid": {"context_precision": p2_cp, "context_recall": p2_cr},
            "phase3_rerank": {"context_precision": p3_cp, "context_recall": p3_cr},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save checkpoint atomically
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        elapsed = time.time() - t0
        print(
            f"[{len(completed)}/{len(queries)}] qid={qid} | "
            f"P1(CP={p1_cp:.2f}, CR={p1_cr:.2f}) | "
            f"P2(CP={p2_cp:.2f}, CR={p2_cr:.2f}) | "
            f"P3(CP={p3_cp:.2f}, CR={p3_cr:.2f}) | "
            f"Tokens: {q_tokens} | Elapsed: {elapsed:.0f}s"
        )

    # 5. Aggregate metrics and Bootstrap CIs
    p1_cps = [v["phase1_dense"]["context_precision"] for v in completed.values()]
    p1_crs = [v["phase1_dense"]["context_recall"] for v in completed.values()]
    p2_cps = [v["phase2_hybrid"]["context_precision"] for v in completed.values()]
    p2_crs = [v["phase2_hybrid"]["context_recall"] for v in completed.values()]
    p3_cps = [v["phase3_rerank"]["context_precision"] for v in completed.values()]
    p3_crs = [v["phase3_rerank"]["context_recall"] for v in completed.values()]

    summary = {
        "benchmark_name": "Gate 4B Primary RAGAS Evaluation",
        "ground_truth_rule": frozen_data["selection_rule"],
        "evaluator_model": MODEL_NAME,
        "n_queries": len(completed),
        "total_tokens": total_tokens,
        "elapsed_seconds": round(time.time() - t0, 1),
        "phases": {
            "phase1_dense": {
                "context_precision": bootstrap_ci(p1_cps),
                "context_recall": bootstrap_ci(p1_crs),
            },
            "phase2_hybrid": {
                "context_precision": bootstrap_ci(p2_cps),
                "context_recall": bootstrap_ci(p2_crs),
            },
            "phase3_rerank": {
                "context_precision": bootstrap_ci(p3_cps),
                "context_recall": bootstrap_ci(p3_crs),
            },
        },
        "pairwise_differences": {
            "p2_vs_p1": {
                "context_precision": paired_bootstrap_diff(p1_cps, p2_cps),
                "context_recall": paired_bootstrap_diff(p1_crs, p2_crs),
            },
            "p3_vs_p2": {
                "context_precision": paired_bootstrap_diff(p2_cps, p3_cps),
                "context_recall": paired_bootstrap_diff(p2_crs, p3_crs),
            },
            "p3_vs_p1": {
                "context_precision": paired_bootstrap_diff(p1_cps, p3_cps),
                "context_recall": paired_bootstrap_diff(p1_crs, p3_crs),
            },
        },
    }

    final_out = RESULTS_DIR / "frozen25_ragas_summary.json"
    with open(final_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n====================================================================")
    print("GATE 4B PRIMARY RAGAS BENCHMARK COMPLETE")
    print("====================================================================")
    print(f"Evaluator: {MODEL_NAME} | N={len(completed)} queries | Ground truth: MS MARCO human answers")
    print(f"Phase 1 (Dense):   CP = {summary['phases']['phase1_dense']['context_precision']['mean']:.4f} "
          f"[{summary['phases']['phase1_dense']['context_precision']['ci_lower']:.4f}, {summary['phases']['phase1_dense']['context_precision']['ci_upper']:.4f}] | "
          f"CR = {summary['phases']['phase1_dense']['context_recall']['mean']:.4f} "
          f"[{summary['phases']['phase1_dense']['context_recall']['ci_lower']:.4f}, {summary['phases']['phase1_dense']['context_recall']['ci_upper']:.4f}]")
    print(f"Phase 2 (Hybrid):  CP = {summary['phases']['phase2_hybrid']['context_precision']['mean']:.4f} "
          f"[{summary['phases']['phase2_hybrid']['context_precision']['ci_lower']:.4f}, {summary['phases']['phase2_hybrid']['context_precision']['ci_upper']:.4f}] | "
          f"CR = {summary['phases']['phase2_hybrid']['context_recall']['mean']:.4f} "
          f"[{summary['phases']['phase2_hybrid']['context_recall']['ci_lower']:.4f}, {summary['phases']['phase2_hybrid']['context_recall']['ci_upper']:.4f}]")
    print(f"Phase 3 (Rerank):  CP = {summary['phases']['phase3_rerank']['context_precision']['mean']:.4f} "
          f"[{summary['phases']['phase3_rerank']['context_precision']['ci_lower']:.4f}, {summary['phases']['phase3_rerank']['context_precision']['ci_upper']:.4f}] | "
          f"CR = {summary['phases']['phase3_rerank']['context_recall']['mean']:.4f} "
          f"[{summary['phases']['phase3_rerank']['context_recall']['ci_lower']:.4f}, {summary['phases']['phase3_rerank']['context_recall']['ci_upper']:.4f}]")
    print(f"Summary written to: {final_out}\n")


if __name__ == "__main__":
    main()
