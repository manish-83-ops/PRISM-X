"""PRISMX RAGAS Evaluation Suite (Family A Non-LLM and Family B LLM Groq Judge)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any
import numpy as np

from prismx.eval.bootstrap import bootstrap_ci, paired_bootstrap_difference

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAGAS_CACHE_DIR = REPO_ROOT / "data" / "cache" / "ragas_judge_cache"

# ====================================================================
# FAMILY A: Non-LLM Context Precision & Recall (Rank & Exact Overlap)
# ====================================================================

def compute_non_llm_metrics(
    retrieved_contexts: list[str],
    retrieved_ids: list[str],
    gold_ids: set[str],
) -> dict[str, float]:
    """Computes exact offline Non-LLM Context Precision and Context Recall.
    
    Context Precision: evaluates the position/rank of the relevant context in top-k.
    Context Recall: fraction of ground truth passages retrieved in top-k.
    """
    if not retrieved_ids or not gold_ids:
        return {"context_precision": 0.0, "context_recall": 0.0}

    # Precision: Mean reciprocal rank / cumulative precision at rank
    # In standard IR/RAGAS formulation for single-labeled context:
    # Context Precision tracks whether relevant context is prioritized at top ranks.
    hits_at_k = 0
    cum_precision = 0.0
    found_relevant = False

    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in gold_ids:
            hits_at_k += 1
            cum_precision += hits_at_k / rank
            found_relevant = True

    context_precision = (cum_precision / hits_at_k) if hits_at_k > 0 else 0.0
    context_recall = (hits_at_k / len(gold_ids)) if len(gold_ids) > 0 else 0.0

    return {
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
    }

# ====================================================================
# FAMILY B: LLM-Based Context Precision & Recall via Groq Free Tier
# ====================================================================

class GroqJudge:
    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | None = None,
    ):
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.cache_dir = cache_dir or RAGAS_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.model_name = model_name

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                if not self.model_name:
                    # Select strongest free model
                    self.model_name = "llama-3.3-70b-versatile"
            except Exception as e:
                print(f"Warning: Could not initialize Groq client: {e}")

    def _call_groq_cached(self, prompt: str, cache_key: str) -> str:
        cache_file = self.cache_dir / f"{cache_key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        if not self.client:
            raise RuntimeError("GROQ_API_KEY environment variable is missing or Groq client not initialized.")

        # Exponential backoff with jitter on 429/5xx
        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                chat = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=0.0,
                )
                output = chat.choices[0].message.content.strip()
                cache_file.write_text(output, encoding="utf-8")
                return output
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 1.0)
                    print(f"Groq Rate limit hit. Backing off for {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2.0)

        raise RuntimeError("Failed to obtain response from Groq after max retries.")

    def judge_context_precision(
        self,
        question: str,
        context: str,
        reference: str,
        query_id: str,
        system_id: str,
        repeat_idx: int = 0,
    ) -> float:
        """Evaluates whether the retrieved context contains relevant information to answer the question."""
        prompt = (
            f"You are an impartial evaluator for information retrieval systems.\n"
            f"Question: {question}\n"
            f"Reference Ground Truth: {reference}\n"
            f"Retrieved Context: {context}\n\n"
            f"Does the Retrieved Context contain information directly useful for answering the Question?\n"
            f"Reply with exactly one word: 'YES' or 'NO'."
        )
        p_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        cache_key = f"prec_{self.model_name}_{p_hash}_{query_id}_{system_id}_r{repeat_idx}"

        res = self._call_groq_cached(prompt, cache_key)
        return 1.0 if "YES" in res.upper() else 0.0

    def judge_context_recall(
        self,
        question: str,
        all_contexts: list[str],
        reference: str,
        query_id: str,
        system_id: str,
        repeat_idx: int = 0,
    ) -> float:
        """Evaluates whether all key factual statements from the reference can be attributed to the retrieved contexts."""
        joined_contexts = "\n---\n".join(all_contexts)
        prompt = (
            f"You are an impartial evaluator for information retrieval systems.\n"
            f"Question: {question}\n"
            f"Reference Answer: {reference}\n"
            f"Retrieved Contexts:\n{joined_contexts}\n\n"
            f"Can the factual assertions in the Reference Answer be substantiated by the Retrieved Contexts?\n"
            f"Reply with a single float score between 0.0 (none) and 1.0 (fully substantiated), followed by nothing else.\n"
            f"Score:"
        )
        p_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        cache_key = f"rec_{self.model_name}_{p_hash}_{query_id}_{system_id}_r{repeat_idx}"

        res = self._call_groq_cached(prompt, cache_key)
        try:
            val = float(res.split()[0].strip())
            return max(0.0, min(1.0, val))
        except Exception:
            return 1.0 if "1" in res or "YES" in res.upper() else 0.0

def run_ragas_evaluation(
    eval_queries: list[dict[str, Any]],
    system_results: dict[str, Any],
    system_id: str = "phase1",
    enable_llm_judge: bool = True,
) -> dict[str, Any]:
    """Computes Family A (Non-LLM) and Family B (LLM-based) RAGAS metrics on the RAGAS set."""
    judge = None
    has_groq_key = bool(os.environ.get("GROQ_API_KEY"))

    if enable_llm_judge and has_groq_key:
        judge = GroqJudge()

    non_llm_precisions = []
    non_llm_recalls = []
    llm_precisions = []
    llm_recalls = []

    for q_item in eval_queries:
        qid = str(q_item["query_id"])
        query = q_item["query"]
        gold_ids = set(str(g) for g in q_item["gold_passage_ids"])
        reference_text = q_item.get("reference_text", "")

        res_item = system_results.get(qid, {})
        retrieved_ids = res_item.get("retrieved_ids", [])
        retrieved_texts = res_item.get("retrieved_texts", [])

        # Family A: Non-LLM
        f_a = compute_non_llm_metrics(retrieved_texts, retrieved_ids, gold_ids)
        non_llm_precisions.append(f_a["context_precision"])
        non_llm_recalls.append(f_a["context_recall"])

        # Family B: LLM-based
        if judge:
            # Precision: check top-1 retrieved context
            top1_text = retrieved_texts[0] if retrieved_texts else ""
            p_llm = judge.judge_context_precision(query, top1_text, reference_text, qid, system_id)
            r_llm = judge.judge_context_recall(query, retrieved_texts, reference_text, qid, system_id)
            llm_precisions.append(p_llm)
            llm_recalls.append(r_llm)

    # Compute bootstrap CIs
    family_a = {
        "context_precision": bootstrap_ci(non_llm_precisions, n_resamples=10000, seed=42),
        "context_recall": bootstrap_ci(non_llm_recalls, n_resamples=10000, seed=42),
        "raw_scores": {
            "context_precision": non_llm_precisions,
            "context_recall": non_llm_recalls,
        },
    }

    family_b = None
    if llm_precisions:
        family_b = {
            "judge_model": judge.model_name,
            "context_precision": bootstrap_ci(llm_precisions, n_resamples=10000, seed=42),
            "context_recall": bootstrap_ci(llm_recalls, n_resamples=10000, seed=42),
            "raw_scores": {
                "context_precision": llm_precisions,
                "context_recall": llm_recalls,
            },
        }

    return {
        "system_id": system_id,
        "n_queries": len(eval_queries),
        "family_a_non_llm": family_a,
        "family_b_llm": family_b,
    }
