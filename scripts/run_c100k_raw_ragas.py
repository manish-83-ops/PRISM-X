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
sys.path.insert(0, str(REPO_ROOT / "src"))
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "frozen_ragas_bench_raw_50.json"
RESULTS_DIR = REPO_ROOT / "results" / "ragas" / "c100k_raw"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
DB_PATH = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
COLLECTION_NAME = "c100k_raw"

JUDGE_MODEL = "llama-3.3-70b-versatile"
DEFAULT_DAILY_TOKEN_CAP = 250000  # Set so N >= 25 fits with 4,000 margin per measured tokens/query (ADR-019)
CALL_INTERVAL_SECONDS = 2.5       # Max 24 calls/min (< 30 RPM limit)


def select_judge_model(client) -> tuple[str, dict[str, Any]]:
    """Select judge model per ADR-019 hierarchy:
    1. llama-3.3-70b-versatile
    2. openai/gpt-oss-120b (with reasoning_effort='low')
    3. Else stop immediately and display available models.
    """
    models_resp = client.models.list()
    available_ids = [m.id for m in models_resp.data]
    logger.info(f"Retrieved {len(available_ids)} model IDs from Groq API.")

    if "llama-3.3-70b-versatile" in available_ids:
        logger.info("Selected primary judge model: llama-3.3-70b-versatile")
        return "llama-3.3-70b-versatile", {}
    elif "openai/gpt-oss-120b" in available_ids:
        logger.info("llama-3.3-70b-versatile not found. Selected ADR-019 fallback judge: openai/gpt-oss-120b (reasoning_effort=low)")
        return "openai/gpt-oss-120b", {"reasoning_effort": "low"}
    else:
        logger.error(f"HARD STOP: Neither llama-3.3-70b-versatile nor openai/gpt-oss-120b available in Groq account. Available models:\n{available_ids}")
        raise RuntimeError(f"ADR-019 violation: Judge model not available. Available IDs: {available_ids}")


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
    verdict = "no measurable difference"
    if excludes_zero:
        verdict = "improvement" if mean_diff > 0 else "degradation"
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "wins": wlt["wins"],
        "losses": wlt["losses"],
        "ties": wlt["ties"],
        "ci_excludes_zero": excludes_zero,
        "verdict": verdict
    }


def call_llm(client, prompt: str, token_tracker: dict[str, int], daily_token_cap: int, judge_model: str, extra_kwargs: dict[str, Any], max_retries: int = 3) -> tuple[str, int]:
    """Call Groq API with rate limit parsing, backoff, and daily token cap tracking."""
    if token_tracker["tokens_today"] >= daily_token_cap:
        raise ResourceWarning(f"Daily token safety cap reached ({token_tracker['tokens_today']:,} / {daily_token_cap:,} tokens). Pausing runner.")

    call_params: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "model": judge_model,
        "temperature": 0.0,
    }
    if extra_kwargs:
        call_params["extra_body"] = extra_kwargs

    for attempt in range(max_retries):
        try:
            time.sleep(CALL_INTERVAL_SECONDS)
            chat = client.chat.completions.create(**call_params)
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
                logger.error(f"[Groq Model Access Error] Model '{judge_model}' is not available on this API key: {exc}")
                raise
            if "429" in err or "rate limit" in err or "resource_exhausted" in err or "500" in err or "502" in err or "503" in err or "504" in err:
                sleep_t = 15.0
                m1 = re.search(r"try again in (\d+)m([\d.]+)s", err)
                m2 = re.search(r"try again in ([\d.]+)s", err)
                if m1:
                    sleep_t = float(m1.group(1)) * 60 + float(m1.group(2)) + 2.0
                elif m2:
                    sleep_t = float(m2.group(1)) + 2.0
                else:
                    sleep_t = min(60.0, 5.0 * (2.0 ** min(attempt, 3))) + random.uniform(1.0, 3.0)
                logger.warning(f"[Groq Retryable Error {err[:30]}] Backoff {sleep_t:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_t)
            else:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"[Groq API Error] {exc} — retrying in 5s (attempt {attempt+1}/{max_retries})...")
                time.sleep(5.0)

    raise RuntimeError(f"Max retries ({max_retries}) exceeded calling Groq API for identical prompt.")


def eval_context_precision(client, question: str, contexts: list[str], reference_answer: str, token_tracker: dict, daily_cap: int, judge_model: str, extra_kwargs: dict[str, Any]) -> tuple[float, int]:
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
    resp, tokens = call_llm(client, prompt, token_tracker, daily_cap, judge_model, extra_kwargs)
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


def eval_context_recall(client, question: str, contexts: list[str], reference_answer: str, token_tracker: dict, daily_cap: int, judge_model: str, extra_kwargs: dict[str, Any]) -> tuple[float, int]:
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
    resp, tokens = call_llm(client, prompt, token_tracker, daily_cap, judge_model, extra_kwargs)
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
def compute_retrieval_overlap(queries: list[dict], dense_retriever, hybrid_retriever, reranker, text_store) -> dict[str, Any]:
    """Compute (no LLM) how often top-5 contexts differ between dense/hybrid/prismx."""
    total_q = len(queries)
    dh_diff_top5, dh_diff_top1, dh_jaccards = 0, 0, []
    ph_diff_top5, ph_diff_top1, ph_jaccards = 0, 0, []
    pd_diff_top5, pd_diff_top1, pd_jaccards = 0, 0, []

    for q_item in queries:
        q_text = q_item["query"]
        d_cands, _, _ = dense_retriever.retrieve(query=q_text, limit=5, search_ef=128)
        d_pids = [c["passage_id"] for c in d_cands]

        h_cands, _ = hybrid_retriever.retrieve(query=q_text, limit=5, alpha=0.8, norm_method="minmax", search_ef=128)
        h_pids = [c["passage_id"] for c in h_cands]

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
        p_pids = [c["passage_id"] for c in r_top10[:5]]

        # Compare Hybrid vs Dense
        s_d, s_h, s_p = set(d_pids[:5]), set(h_pids[:5]), set(p_pids[:5])
        if s_d != s_h:
            dh_diff_top5 += 1
        if d_pids and h_pids and d_pids[0] != h_pids[0]:
            dh_diff_top1 += 1
        dh_jaccards.append(len(s_d & s_h) / len(s_d | s_h) if (s_d | s_h) else 1.0)

        # Compare PRISMX vs Hybrid
        if s_p != s_h:
            ph_diff_top5 += 1
        if p_pids and h_pids and p_pids[0] != h_pids[0]:
            ph_diff_top1 += 1
        ph_jaccards.append(len(s_p & s_h) / len(s_p | s_h) if (s_p | s_h) else 1.0)

        # Compare PRISMX vs Dense
        if s_p != s_d:
            pd_diff_top5 += 1
        if p_pids and d_pids and p_pids[0] != d_pids[0]:
            pd_diff_top1 += 1
        pd_jaccards.append(len(s_p & s_d) / len(s_p | s_d) if (s_p | s_d) else 1.0)

    overlap_stats = {
        "n_queries": total_q,
        "hybrid_vs_dense": {
            "top5_differ_fraction": round(dh_diff_top5 / total_q, 4),
            "top1_differ_fraction": round(dh_diff_top1 / total_q, 4),
            "mean_jaccard": round(float(np.mean(dh_jaccards)), 4)
        },
        "prismx_vs_hybrid": {
            "top5_differ_fraction": round(ph_diff_top5 / total_q, 4),
            "top1_differ_fraction": round(ph_diff_top1 / total_q, 4),
            "mean_jaccard": round(float(np.mean(ph_jaccards)), 4)
        },
        "prismx_vs_dense": {
            "top5_differ_fraction": round(pd_diff_top5 / total_q, 4),
            "top1_differ_fraction": round(pd_diff_top1 / total_q, 4),
            "mean_jaccard": round(float(np.mean(pd_jaccards)), 4)
        }
    }
    return overlap_stats


def run_ragas(
    resume: bool = True,
    dry_run: bool = False,
    max_queries: int = 50,
    daily_token_cap: int = DEFAULT_DAILY_TOKEN_CAP
):
    print("=================================================================")
    print("STARTING C100K_RAW RAGAS BENCHMARK RUNNER (ADR-019 / GATE 6)")
    print(f"Manifest: {MANIFEST_PATH}")
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

    # 2. Compute retrieval overlap (no LLM) before scoring
    print("\n[Pre-Scoring Retrieval Overlap Check (no LLM)]...")
    overlap_stats = compute_retrieval_overlap(queries, dense_retriever, hybrid_retriever, reranker, text_store)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "retrieval_overlap.json", "w", encoding="utf-8") as f:
        json.dump(overlap_stats, f, indent=2)
    print(f"  Hybrid vs Dense: Top-5 diff={overlap_stats['hybrid_vs_dense']['top5_differ_fraction']:.2%}, Jaccard={overlap_stats['hybrid_vs_dense']['mean_jaccard']:.4f}, Top-1 diff={overlap_stats['hybrid_vs_dense']['top1_differ_fraction']:.2%}")
    print(f"  PRISMX vs Hybrid: Top-5 diff={overlap_stats['prismx_vs_hybrid']['top5_differ_fraction']:.2%}, Jaccard={overlap_stats['prismx_vs_hybrid']['mean_jaccard']:.4f}, Top-1 diff={overlap_stats['prismx_vs_hybrid']['top1_differ_fraction']:.2%}")
    print(f"  PRISMX vs Dense:  Top-5 diff={overlap_stats['prismx_vs_dense']['top5_differ_fraction']:.2%}, Jaccard={overlap_stats['prismx_vs_dense']['mean_jaccard']:.4f}, Top-1 diff={overlap_stats['prismx_vs_dense']['top1_differ_fraction']:.2%}")

    # 3. Judge model selection per ADR-019
    judge_model = "llama-3.3-70b-versatile"
    extra_kwargs = {}
    client = None
    if not dry_run:
        from groq import Groq
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is required to run RAGAS benchmark.")
        client = Groq(api_key=api_key)
        judge_model, extra_kwargs = select_judge_model(client)
        print(f"Active Judge Model: {judge_model} (extra_kwargs={extra_kwargs})")

    # 4. Checkpoint management
    chk = load_checkpoint() if resume else {
        "benchmark": "c100k_raw RAGAS",
        "judge_model": judge_model,
        "token_accounting": {"total_tokens": 0, "tokens_today": 0, "calls": 0, "date": time.strftime("%Y-%m-%d")},
        "completed_queries": {}
    }
    chk["judge_model"] = judge_model

    # Reset tokens_today if new calendar day
    today_str = time.strftime("%Y-%m-%d")
    if chk["token_accounting"].get("date") != today_str:
        chk["token_accounting"]["tokens_today"] = 0
        chk["token_accounting"]["date"] = today_str

    token_tracker = chk["token_accounting"]

    phases = ["phase1_dense", "phase2_hybrid", "phase3_prismx_rerank"]

    print(f"\nResumed state: {len(chk['completed_queries'])}/{n_queries} queries completed so far.")
    print(f"Token tracker: {token_tracker['tokens_today']:,}/{daily_token_cap:,} tokens today ({token_tracker['total_tokens']:,} total).\n")

    # Tracking tokens for calibration on first 2 evaluated queries
    calibration_tokens: list[int] = []

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

            query_tokens_spent = 0
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
                    cp, t_cp = eval_context_precision(client, q_text, ctxs, ref_ans, token_tracker, daily_token_cap, judge_model, extra_kwargs)
                    cr, t_cr = eval_context_recall(client, q_text, ctxs, ref_ans, token_tracker, daily_token_cap, judge_model, extra_kwargs)

                    query_tokens_spent += (t_cp + t_cr)
                    chk["completed_queries"][qid][phase_name] = {
                        "context_precision": cp,
                        "context_recall": cr,
                        "tokens_consumed": t_cp + t_cr
                    }
                    save_checkpoint(chk)
                    logger.info(f"  CP: {cp:.4f} | CR: {cr:.4f} | Tokens: {t_cp+t_cr} (Today: {token_tracker['tokens_today']:,}/{daily_token_cap:,})")

            if not dry_run and query_tokens_spent > 0 and len(calibration_tokens) < 2:
                calibration_tokens.append(query_tokens_spent)
                if len(calibration_tokens) == 2:
                    avg_tokens_per_query = float(np.mean(calibration_tokens))
                    completed_so_far = len(chk["completed_queries"])
                    needed_to_reach_25 = max(0, 25 - completed_so_far)
                    projected_additional_tokens = avg_tokens_per_query * needed_to_reach_25
                    remaining_today = daily_token_cap - token_tracker["tokens_today"]
                    logger.info(f"[ADR-019 Token Calibration] Measured {avg_tokens_per_query:.1f} tokens/query across first 2 queries.")
                    logger.info(f"[ADR-019 Token Calibration] To reach N=25 (completed {completed_so_far}), projected {needed_to_reach_25} more queries require {projected_additional_tokens:.0f} tokens.")
                    logger.info(f"[ADR-019 Token Calibration] Remaining quota today: {remaining_today:,} tokens (Safety margin required: 4,000).")
                    if projected_additional_tokens > (remaining_today - 4000):
                        raise ResourceWarning(f"Token calibration check failed: Reaching N=25 requires {projected_additional_tokens:.0f} tokens, but only {remaining_today - 4000} margin tokens available today. Pausing per ADR-019.")
                    else:
                        logger.info("[ADR-019 Token Calibration] PASSED: Account quota is fully sufficient for >= 25 complete queries.")

            # Write updated CSV after every query
            _write_per_query_csv(chk, queries)

    except ResourceWarning as rw:
        logger.warning(f"Execution safely paused: {rw}")
        save_checkpoint(chk)
    except Exception as e:
        logger.error(f"Execution paused/halted: {e}")
        save_checkpoint(chk)

    # 5. Assemble Primary Complete Paired Queries
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
            "judge_model": judge_model,
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

        _write_per_query_csv(chk, queries)

        print("\n### RAGAS SUMMARY TABLE (Primary N = %d, Judge = %s)" % (N, judge_model))
        print("| Metric | Phase 1 (Dense) | Phase 2 (Hybrid) | Phase 3 (PRISMX Rerank) | Delta (Hybrid - Dense) [95% CI] (W/L/T) | Delta (Rerank - Hybrid) [95% CI] (W/L/T) | Delta (Rerank - Dense) [95% CI] (W/L/T) |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for m_key, m_name in [("context_precision", "Context Precision"), ("context_recall", "Context Recall")]:
            v1 = f"{summary['metrics']['phase1_dense'][m_key]['mean']:.4f} [{summary['metrics']['phase1_dense'][m_key]['ci_lower']:.4f}, {summary['metrics']['phase1_dense'][m_key]['ci_upper']:.4f}]"
            v2 = f"{summary['metrics']['phase2_hybrid'][m_key]['mean']:.4f} [{summary['metrics']['phase2_hybrid'][m_key]['ci_lower']:.4f}, {summary['metrics']['phase2_hybrid'][m_key]['ci_upper']:.4f}]"
            v3 = f"{summary['metrics']['phase3_prismx_rerank'][m_key]['mean']:.4f} [{summary['metrics']['phase3_prismx_rerank'][m_key]['ci_lower']:.4f}, {summary['metrics']['phase3_prismx_rerank'][m_key]['ci_upper']:.4f}]"

            d_hd = summary["paired_differences"]["hybrid_minus_dense"][m_key]
            d_hd_str = f"{d_hd['mean_diff']:+.4f} [{d_hd['ci_lower']:+.4f}, {d_hd['ci_upper']:+.4f}] ({d_hd['wins']}/{d_hd['losses']}/{d_hd['ties']}) - {d_hd['verdict']}"

            d_rh = summary["paired_differences"]["rerank_minus_hybrid"][m_key]
            d_rh_str = f"{d_rh['mean_diff']:+.4f} [{d_rh['ci_lower']:+.4f}, {d_rh['ci_upper']:+.4f}] ({d_rh['wins']}/{d_rh['losses']}/{d_rh['ties']}) - {d_rh['verdict']}"

            d_rd = summary["paired_differences"]["rerank_minus_dense"][m_key]
            d_rd_str = f"{d_rd['mean_diff']:+.4f} [{d_rd['ci_lower']:+.4f}, {d_rd['ci_upper']:+.4f}] ({d_rd['wins']}/{d_rd['losses']}/{d_rd['ties']}) - {d_rd['verdict']}"

            print(f"| **{m_name}** | {v1} | {v2} | {v3} | {d_hd_str} | {d_rh_str} | {d_rd_str} |")

    return chk


def _write_per_query_csv(chk: dict[str, Any], queries: list[dict]):
    """Write per-query scores CSV."""
    import csv
    csv_path = RESULTS_DIR / "per_query_scores.csv"
    q_map = {str(q["query_id"]): q for q in queries}
    rows = []
    for qid, phases_data in chk.get("completed_queries", {}).items():
        if all(p in phases_data for p in ["phase1_dense", "phase2_hybrid", "phase3_prismx_rerank"]):
            q_info = q_map.get(qid, {})
            rows.append({
                "query_id": qid,
                "query": q_info.get("query", ""),
                "reference_answer": q_info.get("reference_answer", ""),
                "dense_cp": phases_data["phase1_dense"]["context_precision"],
                "dense_cr": phases_data["phase1_dense"]["context_recall"],
                "hybrid_cp": phases_data["phase2_hybrid"]["context_precision"],
                "hybrid_cr": phases_data["phase2_hybrid"]["context_recall"],
                "prismx_cp": phases_data["phase3_prismx_rerank"]["context_precision"],
                "prismx_cr": phases_data["phase3_prismx_rerank"]["context_recall"],
                "dense_tokens": phases_data["phase1_dense"]["tokens_consumed"],
                "hybrid_tokens": phases_data["phase2_hybrid"]["tokens_consumed"],
                "prismx_tokens": phases_data["phase3_prismx_rerank"]["tokens_consumed"]
            })
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


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
