# PRISMX BENCH Query Split Usage Log

This document records every execution of the 100-query **BENCH** evaluation split throughout the project lifecycle. In strict compliance with zero-leakage and anti-overfitting evaluation rules, the BENCH split was reserved strictly for post-freeze confirmation runs. No hyperparameter (fusion weight $\alpha$, normalization, candidate depth $K$, sequence length, thread count, or governor budget) was ever tuned or selected using the BENCH split.

---

## Summary of BENCH Evaluations

| Run ID | Gate | Date | Git Commit | Config Name | Config Hash | Evaluated Modes | Primary Quality Metrics (Top Mode) | Headline p95 Latency | Result / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BENCH-01** | Gate 3 | 2026-10-03 | `342a639` | `phase1_dense_baseline` & `phase2_hybrid_optimized` | `3b06508...` & `b23eb0d...` | Dense, Hybrid | Phase 2: Hit@1=0.7500, MRR@10=0.8292, NDCG@5=0.8470, Recall@10=0.9533 | 75.03 ms (Hybrid HTTP) | **PASS Gate 3**. Phase 2 directional lift over Phase 1; bootstrap CIs cross zero. |
| **BENCH-02** | Gate 4A | 2026-10-03 | `a6dd8d6` | `phase3_hybrid_rerank` | `0f106e1...` | Dense, Hybrid, Rerank ($K=30$) | Phase 3: Hit@1=0.7600, MRR@10=0.8354, NDCG@5=0.8497, Recall@10=0.9533 | 460.12 ms (Rerank HTTP) | **FAILED SLA**. Latency exceeded $\le 280$ ms hard ceiling. Config rejected. |
| **BENCH-03** | Gate 4B | 2026-10-03 | `0f22a3c` | `phase3_hybrid_rerank_constrained` | `64e95cabb1a1fd58e1ff021ff16043924b7eef86637d9e4defd1c0b5c7c4d2fd` | Dense, Hybrid, Rerank ($K=10$, 200ms Gov) | Phase 3: Hit@1=0.7700, MRR@10=0.8380, NDCG@5=0.8488, Recall@10=0.9533 | **242.25 ms** (Rerank HTTP) | **PASS Gate 4B**. Meets $\le 250$ ms target. Directional gain not statistically significant. |
| **BENCH-04** | Gate 5.4 | 2026-10-03 | `950bc5d` | `c100k_raw_single_run_bench` | `8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf` | Dense, Hybrid, Rerank ($K=10$, Total Gov 250ms) | Rerank: Hit@1=0.4600, MRR@10=0.6532, NDCG@5=0.7236, Recall@10=0.9700 | Pending idle-machine benchmark | **Scored exactly once**. Rerank-Hybrid MRR@10 CI $[+0.0015, +0.1165]$ excludes 0. |

---

## Detailed Run Logs

### Run 1: Gate 3 Phase 1 vs Phase 2 Evaluation
- **Date:** 2026-10-03
- **Commit:** `342a639` (tag: `gate-3-final`)
- **Queries:** 100 BENCH queries (`data/manifests/split_bench.json`, SHA-256: `fab0fe043df8bc77e09f7d4402bda7d50b0b665a69496ad40b36012a5687d28c`).
- **Configs Evaluated:**
  - `phase1_dense_baseline`: Dense cosine retrieval (`BAAI/bge-small-en-v1.5`, 384 dims, HNSW $M=16, ef=100$).
  - `phase2_hybrid_optimized`: Weighted score fusion ($\alpha=0.8$ dense, $0.2$ BM25 sparse with min-max normalization).
- **Selection Basis:** The hybrid fusion parameters ($\alpha=0.8$, min-max normalization) were selected entirely on the **150 seeded TUNE queries** (`results/phase2/fusion_tuning_tune.json`) prior to scoring BENCH.
- **Results:**
  - Phase 1 Dense: Hit@1 = 0.7300, MRR@10 = 0.8096, NDCG@5 = 0.8337, Recall@10 = 0.9533.
  - Phase 2 Hybrid: Hit@1 = 0.7500, MRR@10 = 0.8292, NDCG@5 = 0.8470, Recall@10 = 0.9533.
  - Bootstrap Significance: Mean diff $\Delta$ MRR@10 = +0.0196, 95% CI $[-0.0110, +0.0525]$. CI crosses zero; neutrally reported as not statistically distinguishable.
- **Artifacts:** `results/phase1/metrics.json`, `results/phase2/metrics.json`.

---

### Run 2: Gate 4A Unconstrained Cross-Encoder Reranker ($K=30$)
- **Date:** 2026-10-03
- **Commit:** `a6dd8d6`
- **Queries:** 100 BENCH queries (`data/manifests/split_bench.json`).
- **Config Evaluated:** `phase3_hybrid_rerank`
  - Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (PyTorch dynamic INT8).
  - Candidate Depth: $K = 30$.
  - Sequence Length: `max_length = 256`.
  - Governor: None.
- **Results:**
  - Hit@1 = 0.7600, MRR@10 = 0.8354, NDCG@5 = 0.8497, Recall@10 = 0.9533.
  - Latency: HTTP uncached p95 = **460.12 ms** (severely violated the $\le 280$ ms hard ceiling).
- **Decision:** Configuration **REJECTED** due to SLA non-compliance. Prompted ADR-013 (optimization roadmap: INT8, sequence length 128, candidate depth reduction, and deadline governor).
- **Artifacts:** `results/phase3/bench_raw_k30.json` (Gate 4A archive).

---

### Run 3: Gate 4B Latency-Constrained Cross-Encoder Reranker ($K=10$, 200ms Governor)
- **Date:** 2026-10-03
- **Commit:** `0f22a3c`
- **Queries:** 100 BENCH queries (`data/manifests/split_bench.json`).
- **Config Evaluated:** `phase3_hybrid_rerank_constrained` (Hash: `64e95cabb1a1fd58e1ff021ff16043924b7eef86637d9e4defd1c0b5c7c4d2fd`).
  - Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` with PyTorch dynamic INT8.
  - Candidate Depth: $K = 10$.
  - Sequence Length: `max_length = 128`.
  - Threads: 8 CPU threads.
  - Governor: `rerank_budget_ms = 200.0` ms.
- **Selection Basis (TUNE ONLY):**
  - Candidate depth sweep $K \in \{5, 8, 10, 15, 20\}$ was evaluated on the **150 seeded TUNE queries** under the strict constraint $\text{p95} \le 250$ ms.
  - Results on TUNE:
    - $K=5$: NDCG@5 = 0.9139, p95 = 163.0 ms
    - $K=8$: NDCG@5 = 0.9229, p95 = 203.0 ms
    - **$K=10$ (WINNER)**: NDCG@5 = **0.9296**, MRR@10 = **0.9211**, Hit@1 = **0.8867**, p95 = **170.3 ms**
    - $K=15$: NDCG@5 = 0.9311, p95 = 269.5 ms (FAILED SLA)
    - $K=20$: NDCG@5 = 0.9336, p95 = 266.4 ms (FAILED SLA)
  - $K=10$ was frozen as the sole candidate configuration before executing BENCH.
- **BENCH Results:**
  - Hit@1 = **0.7700** $[0.6900, 0.8500]$ ($\Delta = +0.0400$ vs Dense)
  - MRR@10 = **0.8380** $[0.7750, 0.8964]$ ($\Delta = +0.0284$ vs Dense)
  - NDCG@5 = **0.8488** $[0.7880, 0.9038]$ ($\Delta = +0.0151$ vs Dense)
  - NDCG@10 = **0.8610** $[0.8056, 0.9104]$ ($\Delta = +0.0174$ vs Dense)
  - Recall@10 = **0.9533** $[0.9100, 0.9900]$ ($\Delta = 0.0000$ vs Dense)
  - Latency: HTTP uncached p50 = **181.59 ms**, p90 = **216.69 ms**, p95 = **242.25 ms**, p99 = **280.51 ms** (**PASS** SLA $\le 250$ ms target and $\le 280$ ms hard limit).
  - Statistical Significance: Paired bootstrap differences (10,000 resamples) for all metrics cross zero (e.g., MRR@10 95% CI $[-0.0195, +0.0774]$). Neutrally reported as not statistically distinguishable.
- **Artifacts:** `results/phase3/metrics.json`, `results/phase3/latency_summary.json`.

---

## Evaluation Integrity Statement
1. The 100 BENCH queries have zero passage or query overlap with the 500 TUNE queries or 500 TEST queries.
2. The BENCH set was evaluated for only two reranker candidate configurations ($K=30$ in Gate 4A, $K=10$ in Gate 4B).
3. The winner ($K=10$) was chosen on TUNE only.
4. No threshold or weight was adjusted after viewing BENCH results.

## Run: c100k_raw BENCH Single-Run Evaluation (Gate 5.4)
- **Date:** 2026-10-03 23:09:35
- **Git Commit:** `950bc5d6dbec421266e9cd78934dba2cac533224`
- **Config Hash:** `8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf`
- **Corpus:** `c100k_raw` (100,008 unique passages, uncurated candidate distribution)
- **Split:** `data/c100k_raw/bench_raw_100.json` (N=100)
- **Declaration:** This BENCH set is brand new and has been scored **exactly once**. Zero prior evaluations were executed against this split.
- **Serving Configuration:** HNSW `search_ef=128`, Fusion $\alpha=0.80$ (min-max, not worse than alternatives), Reranker MiniLM-L-6 INT8 ($K=10$, `total_deadline_ms=250.0`, `rerank_budget_ms=200.0`).
- **Governor Telemetry:** Truncation rate = 5.0%, Exhausted rate = 0.0%, Mean candidates scored = 9.75 / 10.
- **Top-1 Non-Gold Sibling Fraction (Unjudged):** Dense = 48.0%, Hybrid = 51.0%, Hybrid+Rerank = 47.0%.
- **Summary Metrics (Mean [95% Bootstrap CI]):**
  - Dense Baseline: Hit@1 = 0.4200, MRR@10 = 0.6032, NDCG@5 = 0.6694, NDCG@10 = 0.6946, Recall@10 = 0.9800, Recall@50 = 0.9900
  - Hybrid Retrieval: Hit@1 = 0.4100, MRR@10 = 0.5949, NDCG@5 = 0.6582, NDCG@10 = 0.6864, Recall@10 = 0.9700, Recall@50 = 0.9800
  - Hybrid + Rerank: Hit@1 = 0.4600, MRR@10 = 0.6532, NDCG@5 = 0.7236, NDCG@10 = 0.7325, Recall@10 = 0.9700, Recall@50 = 0.9800

- **RAGAS Benchmark Alignment Note (Gate 5.5):** The RAGAS evaluation split (`data/manifests/frozen_ragas_bench_raw_50.json`) uses the first 50 valid-answer queries drawn directly from this same frozen BENCH split, evaluated under the identical canonical configuration hash (`8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf`). Zero new BENCH scorings of any alternative configuration were executed.
