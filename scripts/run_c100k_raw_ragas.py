"""RAGAS Evaluation Runner for c100k_raw Frozen 50 BENCH Queries.
In strict accordance with Gate 5.4 Step 3 & ADR-018:
- Judge: llama-3.3-70b-versatile, temperature=0.0, identical across all phases.
- Metrics: Context Precision and Context Recall only.
- Reference: Human reference answer from MS MARCO (ADR-014 rule).
- Manifest: data/manifests/frozen_ragas_bench_raw_50.json (50 valid-answer BENCH queries).
- Order: Query-major order (Phase 1 Dense, Phase 2 Hybrid, Phase 3 Hybrid+Rerank for each query).
  Guarantees complete paired query triples at any stopping point.
- Checkpointing: Checkpoints after every (query, phase) with full --resume support.
- Rate limits & Safety:
  - Official llama-3.3-70b-versatile free-tier limits: 30 RPM, 1,000 RPD, 12K TPM, 100K TPD.
  - Automatic daily token cutoff at 90% of daily limit (default 90,000 tokens/day).
  - Pacing between calls (minimum 2.5s) to stay strictly under 30 RPM and 12k TPM.
  - Exponential backoff for HTTP 429 response codes with header parsing.
- Dry-run mode: --dry-run / --estimate-only (zero LLM calls).
"""

from __future__ import annotations

import argparse
import json
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ragas_c100k_raw")

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "frozen_ragas_bench_raw_50.json"
RESULTS_DIR = REPO_ROOT / "results" / "ragas" / "c100k_raw"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
DB_PATH = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
COLLECTION_NAME = "c100k_raw"

JUDGE_MODEL = "llama-3.3-70b-versatile"
DEFAULT_DAILY_TOKEN_CAP = 96000  # Updated to 96,000 per Gate 5.5 so >=25 complete queries fit
CALL_INTERVAL_SECONDS = 2.5      # Max 24 calls/min (< 30 RPM limit)


def bootstrap_ci(arr: list[float] | np.ndarray, n_resamples: int = 10000, seed: int = 42) -> dict[str, float]:
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


def count_win_loss_tie(a_vals: list[float], b_vals: list[float], eps: float = 1e-6) -> dict[str, int]:
    wins = sum(1 for a, b in zip(a_vals, b_vals) if b - a > eps)
    losses = sum(1 for a, b in zip(a_vals, b_vals) if a - b > eps)
    ties = len(a_vals) - wins - losses
    return {"wins": wins, "losses": losses, "ties": ties}


def paired_bootstrap_diff(a: list[float], b: list[float], n_resamples: int = 10000, seed: int = 42) -> dict[str, Any]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diffs = arr_b - arr_a
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "wins": 0, "losses": 0, "ties": 0, "ci_excludes_zero": False, "verdict": "N/A"}
    rng = np.random.default_rng(seed)
    boot_diffs = np.mean(rng.choice(diffs, size=(n_resamples, n), replace=True), axis=1)
    lo = float(np.percentile(boot_diffs, 2.5))
    hi = float(np.percentile(boot_diffs, 97.5))
    mean_diff = float(np.mean(diffs))
    wlt = count_win_loss_tie(a, b)
    excludes_zero = bool(lo > 0.0 or hi < 0.0)
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "wins": wlt["wins"],
        "losses": wlt["losses"],
        "ties": wlt["ties"],
        "ci_excludes_zero": excludes_zero,
        "verdict": "Statistically Significant Improvement" if excludes_zero and mean_diff > 0 else ("Statistically Significant Degradation" if excludes_zero and mean_diff < 0 else "Directional, not significant (crosses zero)")
    }


def print_estimate_report():
    print("=================================================================")
    print("REVISED RAGAS LLM EVALUATION ESTIMATE (c100k_raw Frozen 50)")
    print("=================================================================")
    print(f"Judge Model: {JUDGE_MODEL} (Temperature=0.0)")
    print("Target Set: 50 Valid-Answer BENCH Queries (Seed-frozen order)")
    print("Metrics: Context Precision and Context Recall (Reference = Human Answer)")
    print("Phases per Query: 3 (Phase 1 Dense, Phase 2 Hybrid, Phase 3 Hybrid+Rerank)")
    print("\n[Empirical Token Measurement]")
    print("  Measured earlier: 91,157 tokens consumed across 75 evaluations")
    print("  Empirical consumption rate: ~1,215.4 tokens per evaluation")
    print("\n[Official Groq Free-Tier Rate Limits for llama-3.3-70b-versatile]")
    print("  Requests Per Minute (RPM): 30 RPM")
    print("  Requests Per Day (RPD):    1,000 RPD")
    print("  Tokens Per Minute (TPM):   12,000 TPM (12k)")
    print("  Tokens Per Day (TPD):      100,000 tokens/day (100k)")
    print("  Automated Safety Cap:      90,000 tokens/day (90% cutoff)")
    print("\n[Total Workload Estimation]")
    total_evals_50 = 50 * 3
    tokens_50 = total_evals_50 * 1215.426
    days_50 = tokens_50 / 90000.0

    target_evals_25 = 25 * 3
    tokens_25 = target_evals_25 * 1215.426
    days_25 = tokens_25 / 90000.0

    print(f"  Target N=50 Complete Queries (150 evaluations total):")
    print(f"    Expected Total Tokens: ~{tokens_50:,.0f} tokens")
    print(f"    Expected API Calls:    ~300 calls (2 calls per evaluation)")
    print(f"    Expected Days (Free Tier @ 90k/day): {days_50:.2f} days (~2 calendar days)")
    print(f"      Day 1: 24 paired queries (72 evaluations, ~87,510 tokens) -> Auto-stop at 90k cap")
    print(f"      Day 2: 26 paired queries (78 evaluations, ~94,800 tokens) -> Complete 50 queries")
    print(f"\n  Target N=25 Minimum Primary Set (75 evaluations total):")
    print(f"    Expected Total Tokens: ~{tokens_25:,.0f} tokens")
    print(f"    Expected API Calls:    ~150 calls")
    print(f"    Expected Days (Free Tier @ 90k/day): {days_25:.2f} days (~1-2 days)")
    print("=================================================================")


def call_llm(client, prompt: str, token_tracker: dict[str, int], daily_token_cap: int, max_retries: int = 15) -> tuple[str, int]:
    """Call Groq API with rate limit parsing, backoff, and daily token cap tracking."""
    if token_tracker["tokens_today"] >= daily_token_cap:
        raise ResourceWarning(f"Daily token safety cap reached ({token_tracker['tokens_today']:,} / {daily_token_cap:,} tokens). Pausing runner.")

    for attempt in range(max_retries):
        try:
            time.sleep(CALL_INTERVAL_SECONDS)
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=JUDGE_MODEL,
                temperature=0.0,
            )
            text = chat.choices[0].message.content.strip()
            usage = getattr(chat, "usage", None)
            tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

            token_tracker["total_tokens"] += tokens
            token_tracker["tokens_today"] += tokens
            token_tracker["calls"] += 1
            return text, tokens
        except Exception as exc:
            err = str(exc).lower()
            if "not_found" in err or "model_not_found" in err or "does not exist" in err:
                logger.error(f"[Groq Model Access Error] Model '{JUDGE_MODEL}' is not available on this API key: {exc}")
                raise
            if "429" in err or "rate limit" in err or "resource_exhausted" in err:
                sleep_t = 15.0
                m1 = re.search(r"try again in (\d+)m([\d.]+)s", err)
                m2 = re.search(r"try again in ([\d.]+)s", err)
                if m1:
                    sleep_t = float(m1.group(1)) * 60 + float(m1.group(2)) + 2.0
                elif m2:
                    sleep_t = float(m2.group(1)) + 2.0
                else:
                    sleep_t = min(60.0, 5.0 * (2.0 ** min(attempt, 4))) + random.uniform(1.0, 3.0)
                logger.warning(f"[Groq 429 Rate Limit] Pacing backoff {sleep_t:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_t)
            else:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"[Groq API Error] {exc} — retrying in 5s...")
                time.sleep(5.0)

    raise RuntimeError("Max retries exceeded calling Groq API.")


def eval_context_precision(client, question: str, contexts: list[str], reference_answer: str, token_tracker: dict, daily_cap: int) -> tuple[float, int]:
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
    resp, tokens = call_llm(client, prompt, token_tracker, daily_cap)
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


def eval_context_recall(client, question: str, contexts: list[str], reference_answer: str, token_tracker: dict, daily_cap: int) -> tuple[float, int]:
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
    resp, tokens = call_llm(client, prompt, token_tracker, daily_cap)
    try:
        val = float(resp.split()[0].strip())
        score = max(0.0, min(1.0, val))
    except Exception:
        score = 1.0 if ("1" in resp or "yes" in resp.lower()) else 0.0
    return round(score, 4), tokens


def load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.is_file():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read checkpoint: {e}")
    return {
        "benchmark": "c100k_raw RAGAS",
        "judge_model": JUDGE_MODEL,
        "token_accounting": {
            "total_tokens": 0,
            "tokens_today": 0,
            "calls": 0,
            "date": time.strftime("%Y-%m-%d")
        },
        "completed_queries": {},  # qid -> {phase -> {cp, cr, tokens}}
    }


def save_checkpoint(data: dict[str, Any]):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_ragas(
    resume: bool = True,
    dry_run: bool = False,
    max_queries: int = 50,
    daily_token_cap: int = DEFAULT_DAILY_TOKEN_CAP
):
    print("=================================================================")
    print("STARTING C100K_RAW RAGAS BENCHMARK RUNNER")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Judge: {JUDGE_MODEL} (Temperature=0.0)")
    print(f"Order: Query-Major (Phase 1, Phase 2, Phase 3 per query)")
    print(f"Dry-run: {dry_run} | Resume: {resume} | Max Queries: {max_queries}")
    print(f"Daily Token Cap: {daily_token_cap:,} tokens")
    print("=================================================================")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    queries = manifest["queries"][:max_queries]
    n_queries = len(queries)
    print(f"Loaded {n_queries} frozen queries.")

    # 1. Initialize PRISMX Retreival Pipeline to extract exact serving contexts
    print("\nInitializing PRISMX retrieval components...")
    from prismx.index.encoder import DenseEncoder
    from prismx.index.lexical import BM25Tokenizer
    from prismx.index.qdrant_store import QdrantStore
    from prismx.index.text_store import TextStore
    from prismx.retrieve.dense import DenseRetriever
    from prismx.retrieve.hybrid import HybridRetriever
    from prismx.retrieve.rerank import CrossEncoderReranker

    qdrant_store = QdrantStore(host="127.0.0.1", port=6333, collection_name=COLLECTION_NAME)
    encoder = DenseEncoder(model_name="BAAI/bge-small-en-v1.5", embedding_dim=384, max_seq_length=128, torch_threads=8)
    tokenizer = BM25Tokenizer()
    text_store = TextStore(db_path=str(DB_PATH))
    dense_retriever = DenseRetriever(encoder=encoder, qdrant_store=qdrant_store, default_candidate_depth=50)
    hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, tokenizer=tokenizer, qdrant_store=qdrant_store, default_candidate_depth=50, default_alpha=0.8)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", torch_threads=8)

    # 2. Checkpoint management
    chk = load_checkpoint() if resume else {
        "benchmark": "c100k_raw RAGAS",
        "judge_model": JUDGE_MODEL,
        "token_accounting": {"total_tokens": 0, "tokens_today": 0, "calls": 0, "date": time.strftime("%Y-%m-%d")},
        "completed_queries": {}
    }

    # Reset tokens_today if new calendar day
    today_str = time.strftime("%Y-%m-%d")
    if chk["token_accounting"].get("date") != today_str:
        chk["token_accounting"]["tokens_today"] = 0
        chk["token_accounting"]["date"] = today_str

    token_tracker = chk["token_accounting"]

    client = None
    if not dry_run:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is required to run RAGAS benchmark.")
        client = Groq(api_key=api_key)

    phases = ["phase1_dense", "phase2_hybrid", "phase3_prismx_rerank"]

    print(f"\nResumed state: {len(chk['completed_queries'])}/{n_queries} queries completed so far.")
    print(f"Token tracker: {token_tracker['tokens_today']:,}/{daily_token_cap:,} tokens today ({token_tracker['total_tokens']:,} total).\n")

    try:
        for idx, q_item in enumerate(queries):
            qid = str(q_item["query_id"])
            q_text = q_item["query"]
            ref_ans = q_item["reference_answer"]

            # Check if this query is already fully completed across all 3 phases
            if qid in chk["completed_queries"] and all(p in chk["completed_queries"][qid] for p in phases):
                continue

            if qid not in chk["completed_queries"]:
                chk["completed_queries"][qid] = {}

            # Retrieve contexts for all 3 phases (identical to BENCH serving config)
            # 1. Dense (ef=128, top-5 hydrated)
            d_cands, _, _ = dense_retriever.retrieve(query=q_text, limit=5, search_ef=128)
            d_pids = [c["passage_id"] for c in d_cands]
            d_map = text_store.get_passages_by_ids(d_pids)
            dense_contexts = [d_map.get(p, {}).get("text", "") for p in d_pids]

            # 2. Hybrid (alpha=0.8, ef=128, top-5 hydrated)
            h_cands, _ = hybrid_retriever.retrieve(query=q_text, limit=5, alpha=0.8, norm_method="minmax", search_ef=128)
            h_pids = [c["passage_id"] for c in h_cands]
            h_map = text_store.get_passages_by_ids(h_pids)
            hybrid_contexts = [h_map.get(p, {}).get("text", "") for p in h_pids]

            # 3. Hybrid + Rerank K=10 (top 10 hydrated, reranked, top 5 selected)
            h10_cands, _ = hybrid_retriever.retrieve(query=q_text, limit=10, alpha=0.8, norm_method="minmax", search_ef=128)
            h10_pids = [c["passage_id"] for c in h10_cands]
            h10_map = text_store.get_passages_by_ids(h10_pids)
            rerank_in = [{"passage_id": p, "text": h10_map.get(p, {}).get("text", ""), "score": c["score"]} for p, c in zip(h10_pids, h10_cands)]

            t_rerank_start = time.perf_counter()
            r_top10, _, _ = reranker.rerank(
                query=q_text,
                candidates=rerank_in,
                top_k=5,
                max_length=128,
                deadline_ms=200.0,
                t_request_start=t_rerank_start,
                batch_size=5
            )
            rerank_contexts = [c.get("text", "") for c in r_top10[:5]]

            contexts_by_phase = {
                "phase1_dense": dense_contexts,
                "phase2_hybrid": hybrid_contexts,
                "phase3_prismx_rerank": rerank_contexts
            }

            # Evaluate phases in query-major order
            for phase_name in phases:
                if phase_name in chk["completed_queries"][qid]:
                    continue

                ctxs = contexts_by_phase[phase_name]

                if dry_run:
                    chk["completed_queries"][qid][phase_name] = {
                        "context_precision": 1.0,
                        "context_recall": 1.0,
                        "tokens_consumed": 0,
                        "mode": "dry_run"
                    }
                    print(f"  [DRY-RUN] Query {idx+1}/{n_queries} (QID {qid}) | {phase_name} verified ({len(ctxs)} contexts).")
                else:
                    logger.info(f"Evaluating Query {idx+1}/{n_queries} (QID {qid}) | Phase: {phase_name}...")
                    cp, t_cp = eval_context_precision(client, q_text, ctxs, ref_ans, token_tracker, daily_token_cap)
                    cr, t_cr = eval_context_recall(client, q_text, ctxs, ref_ans, token_tracker, daily_token_cap)

                    chk["completed_queries"][qid][phase_name] = {
                        "context_precision": cp,
                        "context_recall": cr,
                        "tokens_consumed": t_cp + t_cr
                    }
                    save_checkpoint(chk)
                    logger.info(f"  CP: {cp:.4f} | CR: {cr:.4f} | Tokens: {t_cp+t_cr} (Today: {token_tracker['tokens_today']:,}/{daily_token_cap:,})")

    except ResourceWarning as rw:
        logger.warning(f"Execution safely paused: {rw}")
        save_checkpoint(chk)
    except Exception as e:
        logger.error(f"Execution paused/halted: {e}")
        save_checkpoint(chk)

    # 3. Assemble Primary Complete Paired Queries
    complete_qids = [
        qid for qid, p_map in chk["completed_queries"].items()
        if all(p in p_map for p in phases)
    ]
    N = len(complete_qids)

    print("\n=================================================================")
    print(f"RAGAS EVALUATION CHECKPOINT REACHED")
    print(f"Complete Paired Queries (Primary N): {N}")
    print(f"Tokens Consumed Today: {token_tracker['tokens_today']:,} / {daily_token_cap:,}")
    print(f"Total Tokens Consumed: {token_tracker['total_tokens']:,}")
    print(f"Total API Calls:       {token_tracker['calls']}")
    print("=================================================================")

    if N > 0:
        p1_cp = [chk["completed_queries"][q]["phase1_dense"]["context_precision"] for q in complete_qids]
        p1_cr = [chk["completed_queries"][q]["phase1_dense"]["context_recall"] for q in complete_qids]

        p2_cp = [chk["completed_queries"][q]["phase2_hybrid"]["context_precision"] for q in complete_qids]
        p2_cr = [chk["completed_queries"][q]["phase2_hybrid"]["context_recall"] for q in complete_qids]

        p3_cp = [chk["completed_queries"][q]["phase3_prismx_rerank"]["context_precision"] for q in complete_qids]
        p3_cr = [chk["completed_queries"][q]["phase3_prismx_rerank"]["context_recall"] for q in complete_qids]

        summary = {
            "benchmark": "c100k_raw RAGAS LLM Evaluation",
            "judge_model": JUDGE_MODEL,
            "judge_temperature": 0.0,
            "primary_n_complete_queries": N,
            "target_n": 50,
            "minimum_target_n": 25,
            "selection_bias_caveat": "Queries drawn strictly from MS MARCO validation queries with valid human answers (>1 word, != No Answer Present). RAGAS aggregates exclude queries without human answers.",
            "token_accounting": token_tracker,
            "metrics": {
                "phase1_dense": {
                    "context_precision": bootstrap_ci(p1_cp),
                    "context_recall": bootstrap_ci(p1_cr)
                },
                "phase2_hybrid": {
                    "context_precision": bootstrap_ci(p2_cp),
                    "context_recall": bootstrap_ci(p2_cr)
                },
                "phase3_prismx_rerank": {
                    "context_precision": bootstrap_ci(p3_cp),
                    "context_recall": bootstrap_ci(p3_cr)
                }
            },
            "paired_differences": {
                "hybrid_minus_dense": {
                    "context_precision": paired_bootstrap_diff(p1_cp, p2_cp),
                    "context_recall": paired_bootstrap_diff(p1_cr, p2_cr)
                },
                "rerank_minus_hybrid": {
                    "context_precision": paired_bootstrap_diff(p2_cp, p3_cp),
                    "context_recall": paired_bootstrap_diff(p2_cr, p3_cr)
                },
                "rerank_minus_dense": {
                    "context_precision": paired_bootstrap_diff(p1_cp, p3_cp),
                    "context_recall": paired_bootstrap_diff(p1_cr, p3_cr)
                }
            }
        }

        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n### RAGAS SUMMARY TABLE (Primary N = %d, Judge = %s)" % (N, JUDGE_MODEL))
        print("| Metric | Phase 1 (Dense) | Phase 2 (Hybrid) | Phase 3 (PRISMX Rerank) | Delta (Hybrid - Dense) [95% CI] (W/L/T) | Delta (Rerank - Hybrid) [95% CI] (W/L/T) | Delta (Rerank - Dense) [95% CI] (W/L/T) |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for m_key, m_name in [("context_precision", "Context Precision"), ("context_recall", "Context Recall")]:
            v1 = f"{summary['metrics']['phase1_dense'][m_key]['mean']:.4f} [{summary['metrics']['phase1_dense'][m_key]['ci_lower']:.4f}, {summary['metrics']['phase1_dense'][m_key]['ci_upper']:.4f}]"
            v2 = f"{summary['metrics']['phase2_hybrid'][m_key]['mean']:.4f} [{summary['metrics']['phase2_hybrid'][m_key]['ci_lower']:.4f}, {summary['metrics']['phase2_hybrid'][m_key]['ci_upper']:.4f}]"
            v3 = f"{summary['metrics']['phase3_prismx_rerank'][m_key]['mean']:.4f} [{summary['metrics']['phase3_prismx_rerank'][m_key]['ci_lower']:.4f}, {summary['metrics']['phase3_prismx_rerank'][m_key]['ci_upper']:.4f}]"

            d_hd = summary["paired_differences"]["hybrid_minus_dense"][m_key]
            d_hd_str = f"{d_hd['mean_diff']:+.4f} [{d_hd['ci_lower']:+.4f}, {d_hd['ci_upper']:+.4f}] ({d_hd['wins']}/{d_hd['losses']}/{d_hd['ties']})"

            d_rh = summary["paired_differences"]["rerank_minus_hybrid"][m_key]
            d_rh_str = f"{d_rh['mean_diff']:+.4f} [{d_rh['ci_lower']:+.4f}, {d_rh['ci_upper']:+.4f}] ({d_rh['wins']}/{d_rh['losses']}/{d_rh['ties']})"

            d_rd = summary["paired_differences"]["rerank_minus_dense"][m_key]
            d_rd_str = f"{d_rd['mean_diff']:+.4f} [{d_rd['ci_lower']:+.4f}, {d_rd['ci_upper']:+.4f}] ({d_rd['wins']}/{d_rd['losses']}/{d_rd['ties']})"

            print(f"| **{m_name}** | {v1} | {v2} | {v3} | {d_hd_str} | {d_rh_str} | {d_rd_str} |")

    return chk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS benchmark on c100k_raw frozen 50 BENCH queries.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoint.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not resume; start fresh.")
    parser.add_argument("--dry-run", action="store_true", help="Test context retrieval and prompting without calling Groq.")
    parser.add_argument("--estimate-only", action="store_true", help="Print token/day estimation report and exit.")
    parser.add_argument("--max-queries", type=int, default=50, help="Maximum queries to evaluate (default 50).")
    parser.add_argument("--daily-token-cap", type=int, default=DEFAULT_DAILY_TOKEN_CAP, help="Daily token cutoff (default 90000).")
    args = parser.parse_args()

    if args.estimate_only:
        print_estimate_report()
    else:
        run_ragas(
            resume=args.resume,
            dry_run=args.dry_run,
            max_queries=args.max_queries,
            daily_token_cap=args.daily_token_cap
        )
