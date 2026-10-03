"""Parity Gate Verification for Cross-Encoder Runtime Variants (ADR-022 / Part 1.4).
Pre-registered criteria against PyTorch FP32 CE across all 500 TUNE queries:
1. Top-1 agreement >= 0.97
2. Top-5 set agreement >= 0.98
3. Paired delta NDCG@5 with |delta| <= 0.005 and 95% CI containing 0
4. Paired delta Hit@1 with |delta| <= 0.005 and 95% CI containing 0
Variants evaluated:
- ONNX FP32 (max_length=128)
- ONNX INT8 (max_length=128)
- ONNX INT8 (max_length=96)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prismx.eval.metrics import hit_at_1, ndcg_at_k
from prismx.index.text_store import TextStore

TUNE_RETRIEVALS_PATH = REPO_ROOT / "results" / "tune_per_query_retrievals.json"
ONNX_FP32_PATH = REPO_ROOT / "models" / "onnx_cross_encoder" / "model_fp32.onnx"
ONNX_INT8_PATH = REPO_ROOT / "models" / "onnx_cross_encoder" / "model_int8.onnx"
DB_PATH = REPO_ROOT / "data" / "c100k_raw" / "text_store_raw.db"
OUT_REPORT = REPO_ROOT / "results" / "parity_gates_report.json"


def paired_bootstrap(diffs: np.ndarray, n_resamples: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.mean(rng.choice(diffs, size=(n_resamples, n), replace=True), axis=1)
    mean_diff = float(np.mean(diffs))
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return round(mean_diff, 5), round(low, 5), round(high, 5)


def run_pytorch_ce(model, tokenizer, pairs, max_length=128) -> list[float]:
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)
        if logits.dim() == 0:
            return [float(logits.item())]
        return logits.tolist()


def run_onnx_ce(session, tokenizer, pairs, max_length=128) -> list[float]:
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"],
    }
    logits = session.run(None, ort_inputs)[0].squeeze(-1)
    if logits.ndim == 0:
        return [float(logits.item())]
    return logits.tolist()


def main():
    print("=" * 70)
    print("PARITY GATES EVALUATION ON ALL 500 TUNE QUERIES (ADR-022 / PART 1.4)")
    print("=" * 70)

    # 1. Load TUNE queries and first-stage hybrid pools
    with open(TUNE_RETRIEVALS_PATH, "r", encoding="utf-8") as f:
        tune_data = json.load(f)
    print(f"Loaded {len(tune_data)} TUNE queries.")

    text_store = TextStore(db_path=str(DB_PATH))

    # Pre-gather all candidate texts for the top-10 hybrid passages
    all_needed_pids = set()
    for q in tune_data:
        all_needed_pids.update(q["hybrid_pids"][:10])
    print(f"Hydrating {len(all_needed_pids)} candidate passages from SQLite...")
    passage_map = text_store.get_passages_by_ids(list(all_needed_pids))

    # 2. Initialize Models
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading PyTorch FP32 Baseline: {model_name}...")
    torch_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    torch_model.eval()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 6
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"Loading ONNX FP32 Session from {ONNX_FP32_PATH}...")
    session_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), sess_opts, providers=["CPUExecutionProvider"])

    print(f"Loading ONNX INT8 Session from {ONNX_INT8_PATH}...")
    session_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), sess_opts, providers=["CPUExecutionProvider"])

    # 3. Variants to compare against PyTorch FP32 Baseline
    variants = [
        ("onnx_fp32_ml128", session_fp32, 128),
        ("onnx_int8_ml128", session_int8, 128),
        ("onnx_int8_ml96",  session_int8, 96),
    ]

    # Pre-allocate rankings
    baseline_rankings = []
    baseline_hit1 = []
    baseline_ndcg5 = []

    variant_results = {
        name: {
            "rankings": [],
            "hit1": [],
            "ndcg5": [],
        } for name, _, _ in variants
    }

    t0 = time.time()
    print("\nReranking 500 queries across PyTorch baseline and ONNX variants...")
    for idx, q in enumerate(tune_data):
        query = q["query"]
        gold_ids = set(q["gold_pids"])
        top10_pids = q["hybrid_pids"][:10]
        pairs = [[query, passage_map.get(pid, {}).get("text", "")] for pid in top10_pids]

        # Baseline: PyTorch FP32 max_length=128
        scores_base = run_pytorch_ce(torch_model, tokenizer, pairs, max_length=128)
        base_sorted_pids = [pid for _, pid in sorted(zip(scores_base, top10_pids), key=lambda x: x[0], reverse=True)]
        baseline_rankings.append(base_sorted_pids)
        baseline_hit1.append(hit_at_1(base_sorted_pids, gold_ids))
        baseline_ndcg5.append(ndcg_at_k(base_sorted_pids, gold_ids, 5))

        # Variants
        for name, sess, max_len in variants:
            scores_var = run_onnx_ce(sess, tokenizer, pairs, max_length=max_len)
            var_sorted_pids = [pid for _, pid in sorted(zip(scores_var, top10_pids), key=lambda x: x[0], reverse=True)]
            variant_results[name]["rankings"].append(var_sorted_pids)
            variant_results[name]["hit1"].append(hit_at_1(var_sorted_pids, gold_ids))
            variant_results[name]["ndcg5"].append(ndcg_at_k(var_sorted_pids, gold_ids, 5))

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/500 queries ({time.time() - t0:.1f}s)...")

    # 4. Evaluate Pre-Registered Parity Gates
    print("\n" + "=" * 70)
    print("PARITY GATE DECISION REPORT")
    print("=" * 70)

    n_q = len(tune_data)
    report_data = {
        "n_queries": n_q,
        "baseline": {
            "name": "pytorch_fp32_ml128",
            "mean_hit1": round(float(np.mean(baseline_hit1)), 4),
            "mean_ndcg5": round(float(np.mean(baseline_ndcg5)), 4),
        },
        "variants": {}
    }

    base_hit1_arr = np.array(baseline_hit1)
    base_ndcg5_arr = np.array(baseline_ndcg5)

    for name, _, max_len in variants:
        v_ranks = variant_results[name]["rankings"]
        v_hit1_arr = np.array(variant_results[name]["hit1"])
        v_ndcg5_arr = np.array(variant_results[name]["ndcg5"])

        # Metric 1: Top-1 Agreement >= 0.97
        top1_agrees = sum(1 for b, v in zip(baseline_rankings, v_ranks) if b[0] == v[0])
        top1_agree_rate = top1_agrees / n_q

        # Metric 2: Top-5 Set Agreement >= 0.98
        top5_jaccards = [len(set(b[:5]).intersection(set(v[:5]))) / len(set(b[:5]).union(set(v[:5]))) for b, v in zip(baseline_rankings, v_ranks)]
        mean_top5_jaccard = float(np.mean(top5_jaccards))

        # Metric 3: Delta NDCG@5 with |delta| <= 0.005 and CI containing 0
        diff_ndcg5 = v_ndcg5_arr - base_ndcg5_arr
        m_ndcg5, lo_ndcg5, hi_ndcg5 = paired_bootstrap(diff_ndcg5)

        # Metric 4: Delta Hit@1 with |delta| <= 0.005 and CI containing 0
        diff_hit1 = v_hit1_arr - base_hit1_arr
        m_hit1, lo_hit1, hi_hit1 = paired_bootstrap(diff_hit1)

        pass_top1 = top1_agree_rate >= 0.97
        pass_top5 = mean_top5_jaccard >= 0.98
        pass_ndcg5 = abs(m_ndcg5) <= 0.005 and (lo_ndcg5 <= 0.0 <= hi_ndcg5)
        pass_hit1 = abs(m_hit1) <= 0.005 and (lo_hit1 <= 0.0 <= hi_hit1)

        all_pass = pass_top1 and pass_top5 and pass_ndcg5 and pass_hit1

        status_str = "PASS (ADOPT)" if all_pass else "FAIL (DO NOT ADOPT)"

        print(f"\n--- Variant: {name} [{status_str}] ---")
        print(f"  Top-1 Agreement        : {top1_agree_rate:.4f}  (Gate: >= 0.97 -> {'PASS' if pass_top1 else 'FAIL'})")
        print(f"  Top-5 Set Agreement    : {mean_top5_jaccard:.4f}  (Gate: >= 0.98 -> {'PASS' if pass_top5 else 'FAIL'})")
        print(f"  Delta NDCG@5           : {m_ndcg5:+.5f} [{lo_ndcg5:+.5f}, {hi_ndcg5:+.5f}]  (Gate: |delta|<=0.005 & CI contains 0 -> {'PASS' if pass_ndcg5 else 'FAIL'})")
        print(f"  Delta Hit@1            : {m_hit1:+.5f} [{lo_hit1:+.5f}, {hi_hit1:+.5f}]  (Gate: |delta|<=0.005 & CI contains 0 -> {'PASS' if pass_hit1 else 'FAIL'})")
        print(f"  Overall Parity Gate    : {status_str}")

        report_data["variants"][name] = {
            "max_length": max_len,
            "top1_agreement": round(top1_agree_rate, 4),
            "pass_top1": pass_top1,
            "top5_set_agreement": round(mean_top5_jaccard, 4),
            "pass_top5": pass_top5,
            "delta_ndcg5": m_ndcg5,
            "ndcg5_ci_lower": lo_ndcg5,
            "ndcg5_ci_upper": hi_ndcg5,
            "pass_ndcg5": pass_ndcg5,
            "delta_hit1": m_hit1,
            "hit1_ci_lower": lo_hit1,
            "hit1_ci_upper": hi_hit1,
            "pass_hit1": pass_hit1,
            "gate_passed": all_pass,
        }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved parity gate verification report to {OUT_REPORT}")

if __name__ == "__main__":
    main()
