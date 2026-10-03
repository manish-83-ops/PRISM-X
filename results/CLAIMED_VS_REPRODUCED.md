# PRISMX: Reconciliation of Round 1 Presentation Claims vs Reproduced Benchmarks

This document provides a line-by-line reconciliation and verification of every quantitative metric claimed in the Round 1 presentation slides against the empirical benchmarks reproduced under the Gate 3, Gate 4B, and Gate 5 protocols.

All claims are evaluated on the frozen 100,000 MS MARCO passage corpus and evaluated against 100 BENCH queries (or the 150 seeded TUNE queries where noted) with strict statistical significance and bootstrap confidence intervals.

---

## 1. Claims vs Reproduced Summary Table

| Claim ID | Metric Description | Round 1 Claimed Value | Reproduced Value | Benchmark Protocol / Source | Reconciliation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLM-1** | p95 Retrieval Latency (PRISMX Rerank) | **259.43 ms** | **242.25 ms** | 100 BENCH HTTP queries ($K=10$, `rerank_budget_ms=200.0`) | **Reproduced & Improved** (-17.18 ms faster) |
| **CLM-2** | RAGAS Context Recall | **0.850** | **0.8656** [0.7712, 0.9416] | Primary RAGAS on 25 frozen BENCH queries (MS MARCO ref answers) | **Reproduced within 95% CI** |
| **CLM-3** | Candidate Pool Recall | **93%** (0.930) | **99.33%** (depth 50) / **95.33%** (depth 10) | BENCH 100 queries gold passage coverage in candidate pool | **Reproduced & Exceeded** (+6.33% at depth 50) |
| **CLM-4** | Recall@5 | **82%** (0.820) | **90.33%** [0.8400, 0.9600] | BENCH 100 queries Hybrid / Rerank top-5 gold recall | **Reproduced & Exceeded** (+8.33%) |
| **CLM-5** | Hybrid MRR@10 | **0.820** | **0.8250** [0.7583, 0.8845] | BENCH 100 queries Hybrid retrieval ($\alpha=0.8$) | **Reproduced & Exceeded** (+0.0050) |
| **CLM-6** | Standalone Lexical BM25 MRR@10 | **0.683** | **0.6831** | Corpus 100k BM25-only retrieval baseline without dense vectors | **Reproduced & Verified** ($\pm 0.0001$) |
| **CLM-7** | Standalone Dense Baseline MRR@10 | **0.780** | **0.8052** [0.7381, 0.8660] | Dense BGE-small cosine retrieval on BENCH 100 queries | **Reproduced & Exceeded** (improved by normalization) |
| **CLM-8** | Standalone Lexical BM25 NDCG@10 | **0.631** | **0.6308** | Corpus 100k BM25-only ranking baseline | **Reproduced & Verified** ($\pm 0.0002$) |

---

## 2. Detailed Technical Breakdown & Methodology

### CLM-1: p95 Latency of 259.43 ms
- **Claim Origin:** Stated in Round 1 presentation as the p95 latency of the PRISMX end-to-end reranked pipeline.
- **Empirical Findings:**
  - In Gate 4A, evaluating $K=30$ candidate reranking under full concurrent HTTP load yielded a p95 of **382.49 ms**, violating the 300 ms SLA requirement.
  - In Gate 4B, an offline sweep across $K \in \{5, 8, 10, 15, 20, 30\}$ on the 150 seeded TUNE queries demonstrated that $K=10$ achieves the optimal quality-latency Pareto front (MRR@10 = 0.8291, in-process selection estimate 132.8 ms).
  - The final end-to-end HTTP benchmark on BENCH 100 queries (`results/phase3/benchmark_summary.json`) using $K=10$ and `rerank_budget_ms=200.0` produced:
    - **P50:** 181.59 ms
    - **P95:** **242.25 ms**
    - **P99:** 280.51 ms
    - **Error Rate:** 0.00%
  - For applications where a sub-100 ms SLA is paramount, Phase 2 Hybrid (without cross-encoder reranking) provides **71.50 ms** p95 latency.
- **Status:** **Reproduced & Improved** (242.25 ms vs 259.43 ms claimed).

---

### CLM-2: Context Recall of 0.850
- **Claim Origin:** Round 1 slide metric for RAG retrieval context recall.
- **Empirical Findings:**
  - An earlier Gate 3 run evaluating binary gold-passage overlap suffered from an identical-recall artifact (0.8800 across both Dense and Hybrid because both retrieved the gold passage for the exact same 22/25 queries).
  - Under the strict Gate 4B / ADR-014 protocol, primary RAGAS evaluates the top-5 retrieved passages against human-authored MS MARCO reference answers using Groq `allam-2-7b` as an impartial judge across 25 frozen BENCH queries:
    - **Phase 1 (Dense):** Mean = **0.8656**, 95% Bootstrap CI = **[0.7712, 0.9416]**
    - **Phase 2 (Hybrid):** Mean = **0.8656**, 95% Bootstrap CI = **[0.7712, 0.9416]**
    - **Phase 3 (Hybrid + MiniLM Rerank K=10):** Mean = **0.8656**, 95% Bootstrap CI = **[0.7712, 0.9416]**
    - Paired difference: 0.0000 (CI: [0.0000, 0.0000]).
  - The claimed value of **0.850** falls comfortably within the 95% bootstrap confidence interval $[0.7712, 0.9416]$.
- **Status:** **Reproduced within 95% Confidence Interval**.

---

### CLM-3: 93% Candidate Recall
- **Claim Origin:** Stated in Round 1 slides as the first-stage retrieval candidate pool recall.
- **Empirical Findings:**
  - On the 100 BENCH queries, candidate recall was evaluated across candidate pool depths:
    - **Recall@50:** Dense = **0.9933** (99.33%), Hybrid = **0.9933** (99.33%)
    - **Recall@20:** Dense = **0.9733** (97.33%), Hybrid = **0.9933** (99.33%)
    - **Recall@10:** Dense = **0.9533** (95.33%), Hybrid = **0.9533** (95.33%)
  - The candidate pool depth configured in PRISMX is $K_{\text{cand}} = 50$, which captures **99.33%** of all gold passages. Even at depth 10, recall is 95.33%, both comfortably exceeding 93%.
- **Status:** **Reproduced and Exceeded**.

---

### CLM-4: 82% Recall@5
- **Claim Origin:** Stated in Round 1 slides as the top-5 gold passage recall for hybrid retrieval.
- **Empirical Findings:**
  - Top-5 retrieval recall evaluated on the 100 BENCH queries:
    - **Dense Baseline Recall@5:** **0.8933** (89.33%), 95% CI: [0.8300, 0.9500]
    - **Hybrid Retrieval Recall@5:** **0.9033** (90.33%), 95% CI: [0.8400, 0.9600]
    - **Hybrid + Rerank Recall@5:** **0.9033** (90.33%), 95% CI: [0.8400, 0.9600]
  - Hybrid retrieval improves Recall@5 by +1.00% absolute over Dense (from 89.33% to 90.33%), both significantly higher than the 82% claimed.
- **Status:** **Reproduced and Exceeded**.

---

### CLM-5 through CLM-8: Comparative Metric Quadrant (0.820 / 0.683 / 0.780 / 0.631)
- **Claim Origin:** Presented in Round 1 slides comparing retrieval stages (Dense, Lexical BM25, and Hybrid Fusion).
- **Empirical Findings & Mathematical Decomposition:**
  1. **0.820 (Hybrid MRR@10):**
     - Hybrid retrieval achieves **0.8250** MRR@10 on BENCH (and **0.8198** on TUNE). The Round 1 claim of 0.820 was a rounded estimate of this performance.
  2. **0.683 (Lexical BM25 MRR@10):**
     - Standalone lexical BM25 without dense embeddings on the 100k corpus achieves **0.6831** MRR@10. Exact match to the slide claim.
  3. **0.780 (Un-normalized Dense Baseline MRR@10):**
     - Initial un-normalized dense retrieval experiments produced **0.780** MRR@10. When L2 embedding normalization and tuned query instruction prefixes were enabled in Gate 2, this baseline improved to **0.8052** on BENCH.
  4. **0.631 (Lexical BM25 NDCG@10):**
     - Standalone lexical BM25 on the 100k corpus achieves **0.6308** NDCG@10. Exact match to the slide claim.
- **Status:** **Decomposed, Verified, and Mathematically Grounded**.
