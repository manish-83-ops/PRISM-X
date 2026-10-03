"""Gate 4B: Systematically benchmark reranker speedup techniques and log per-pair costs."""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prismx.index.text_store import TextStore

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("====================================================================")
    print("GATE 4B: RERANKER SPEEDUP BENCHMARK & PROFILING")
    print("====================================================================\n")

    # 1. Fetch real passages from SQLite
    text_store = TextStore(db_path=str(REPO_ROOT / "data" / "text_store.db"))
    bench_file = REPO_ROOT / "data" / "manifests" / "split_bench.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    # Use first 20 BENCH queries to measure real speedups
    test_queries = bench_queries[:20]
    # Sample 30 real passages
    sample_pids = [str(q["gold_passage_ids"][0]) for q in bench_queries[:30]]
    hydrated = text_store.get_passages_by_ids(sample_pids)
    sample_texts = [hydrated[pid]["text"] for pid in sample_pids if pid in hydrated][:30]

    q_text = test_queries[0]["query"]
    pairs_10 = [[q_text, t] for t in sample_texts[:10]]
    pairs_20 = [[q_text, t] for t in sample_texts[:20]]
    pairs_30 = [[q_text, t] for t in sample_texts[:30]]

    results_log = {}

    # Experiment 1: Tokenization vs Forward Pass breakdown (MiniLM-L6 FP32)
    tok_l6 = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    m_l6_fp32 = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    m_l6_fp32.eval()

    # Warmup
    inp_warm = tok_l6(pairs_10, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        m_l6_fp32(**inp_warm)

    # Profile K=10, 20, 30
    for k, pairs in [(10, pairs_10), (20, pairs_20), (30, pairs_30)]:
        t0 = time.perf_counter()
        inp = tok_l6(pairs, padding=True, truncation=True, max_length=256, return_tensors="pt")
        t_tok = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        with torch.inference_mode():
            out = m_l6_fp32(**inp)
        t_fwd = (time.perf_counter() - t1) * 1000.0

        per_pair = (t_tok + t_fwd) / k
        print(f"MiniLM-L6 FP32 (max_len=256) K={k}: Total={t_tok+t_fwd:.1f}ms | Tok={t_tok:.1f}ms | Fwd={t_fwd:.1f}ms | Per-pair={per_pair:.2f}ms")
        results_log[f"L6_FP32_K{k}"] = {"tok_ms": t_tok, "fwd_ms": t_fwd, "total_ms": t_tok+t_fwd, "per_pair_ms": per_pair}

    # Experiment 2: FP32 vs Dynamic INT8 Quantization (Linear layers)
    m_l6_int8 = torch.quantization.quantize_dynamic(m_l6_fp32, {torch.nn.Linear}, dtype=torch.qint8)
    m_l6_int8.eval()

    for k, pairs in [(10, pairs_10), (20, pairs_20), (30, pairs_30)]:
        inp = tok_l6(pairs, padding=True, truncation=True, max_length=256, return_tensors="pt")
        t1 = time.perf_counter()
        with torch.inference_mode():
            out_int8 = m_l6_int8(**inp)
        t_fwd_int8 = (time.perf_counter() - t1) * 1000.0
        print(f"MiniLM-L6 INT8 (max_len=256) K={k}: Fwd={t_fwd_int8:.1f}ms (Speedup vs FP32: {results_log[f'L6_FP32_K{k}']['fwd_ms']/t_fwd_int8:.2f}x)")
        results_log[f"L6_INT8_K{k}"] = {"fwd_ms": t_fwd_int8}

    # Experiment 3: Max Length Truncation (256 vs 192 vs 128)
    print("\nMax Length Truncation Sweep (MiniLM-L6 INT8, K=10):")
    for max_l in [256, 192, 128]:
        inp = tok_l6(pairs_10, padding=True, truncation=True, max_length=max_l, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = m_l6_int8(**inp)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  max_length={max_l}: Latency={dt:.1f}ms (Speedup vs 256: {results_log['L6_INT8_K10']['fwd_ms']/dt:.2f}x)")
        results_log[f"L6_INT8_max_len_{max_l}"] = {"latency_ms": dt}

    # Experiment 4: Thread Scaling (4, 6, 8, 12 threads)
    print("\nThread Scaling Sweep (MiniLM-L6 INT8, K=10, max_len=128):")
    for n_th in [4, 6, 8, 12]:
        torch.set_num_threads(n_th)
        inp = tok_l6(pairs_10, padding=True, truncation=True, max_length=128, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = m_l6_int8(**inp)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  threads={n_th}: Latency={dt:.1f}ms")
        results_log[f"L6_INT8_threads_{n_th}"] = {"latency_ms": dt}
    torch.set_num_threads(12)

    # Experiment 5: Candidate Model MiniLM-L4 vs MiniLM-L6
    tok_l4 = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-4-v2")
    m_l4_fp32 = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-4-v2")
    m_l4_int8 = torch.quantization.quantize_dynamic(m_l4_fp32, {torch.nn.Linear}, dtype=torch.qint8)
    m_l4_int8.eval()

    print("\nModel Architecture Comparison (K=10, max_len=128, INT8):")
    for m_name, tok, model in [("MiniLM-L6", tok_l6, m_l6_int8), ("MiniLM-L4", tok_l4, m_l4_int8)]:
        inp = tok(pairs_10, padding=True, truncation=True, max_length=128, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inp)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  {m_name}: Latency={dt:.1f}ms")
        results_log[f"{m_name}_K10_len128"] = {"latency_ms": dt}

    # Experiment 6: Ranking Parity (PyTorch INT8 vs PyTorch FP32 on K=30)
    inp_30_fp32 = tok_l6(pairs_30, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        scores_fp32 = m_l6_fp32(**inp_30_fp32).logits.squeeze(-1).tolist()
        scores_int8 = m_l6_int8(**inp_30_fp32).logits.squeeze(-1).tolist()

    corr, _ = spearmanr(scores_fp32, scores_int8)
    order_fp32 = np.argsort(scores_fp32)[::-1][:5]
    order_int8 = np.argsort(scores_int8)[::-1][:5]
    top5_agree = len(set(order_fp32).intersection(set(order_int8))) / 5.0
    print(f"\nRanking Parity (FP32 vs INT8, K=30): Spearman Rank Corr = {corr:.4f} | Top-5 Agreement = {top5_agree*100:.1f}%")
    results_log["ranking_parity"] = {"spearman_rank_correlation": float(corr), "top5_agreement": top5_agree}

    out_file = REPO_ROOT / "results" / "phase3" / "reranker_speedup_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nSaved speedup benchmark results to {out_file}")

    text_store.close()

if __name__ == "__main__":
    main()
