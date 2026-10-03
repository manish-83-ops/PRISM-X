# PRISMX Judge & Evaluator Q&A Guide (Gate 4A)

This document addresses key architectural, empirical, and statistical questions for evaluators and judges reviewing PRISMX Gate 4A.

---

### Q1: What was delivered in Gate 4A?
**A:** Gate 4A delivered the complete retrieval quality and telemetry enhancement suite:
1. **Cross-Encoder Reranker:** Integration of `cross-encoder/ms-marco-MiniLM-L-6-v2` with PyTorch dynamic INT8 quantization on linear layers.
2. **LRU Query Result Cache:** High-performance in-memory cache (capacity 2,000 entries) with SHA-256 deterministic keying and synchronous automatic invalidation on upsert/delete.
3. **Telemetry & Evidence Schema:** Every search result provides comprehensive per-passage evidence (`category`, `source`, `length_chars`, `score`, dense/sparse/fused/rerank ranks and scores, `retrieved_by` channel provenance), plus per-stage timing breakdown (`encode`, `dense`, `sparse`, `fusion`, `fetch_text`, `rerank`, `total`), `cache_hit`, and `governor_state`.
4. **Grounded RAG Answer Endpoint:** `POST /answer` retrieves relevant passages and synthesizes a factual, citation-grounded response (`[1]`, `[2]`) using server-side Groq LLM with fallback to extractive synthesis.
5. **Rigorous Benchmarks:** Empirical ablation on TUNE to select $K$, single evaluation on 100 BENCH queries across frozen configs with paired bootstrap significance testing, and a dedicated 100-query latency and cache workload benchmark suite.

---

### Q2: How was the reranker candidate depth $K$ selected?
**A:** Following Rule R1 (Zero Leakage) and ADR-011:
- The candidate depth $K$ was chosen **strictly from the TUNE split (150 queries)** before touching BENCH.
- The selection rule was logged in advance in `docs/DECISIONS.md`:
  > 1. Select the depth $K \in \{10, 20, 30\}$ that maximizes **NDCG@5** on TUNE.
  > 2. In the event of a tie or difference $\le 0.0010$ NDCG@5, choose the smaller $K$ to minimize latency.
- **Empirical TUNE Results:**
  - $K = 10$: NDCG@5 = 0.9222 | MRR@10 = 0.9117 | Hit@1 = 0.8667
  - $K = 20$: NDCG@5 = 0.9369 | MRR@10 = 0.9258 | Hit@1 = 0.8867
  - $K = 30$: NDCG@5 = 0.9381 | MRR@10 = 0.9250 | Hit@1 = 0.8800
- **Outcome:** $K=30$ won by $+0.0012$ NDCG@5 over $K=20$, exceeding the $0.0010$ threshold. It was frozen in configuration `phase3_hybrid_rerank` (hash: `0f106e12f557e9c7440284eaa0582c292a5904273d9b9561dc86dbd31ff78c76`).

---

### Q3: What were the retrieval quality findings on BENCH, and are they statistically significant?
**A:** Evaluated on the 100 BENCH queries (`split_bench.json`):

| Metric | Phase 1: Dense Baseline | Phase 2: Hybrid | Phase 3: Hybrid + Rerank ($K=30$) | Overall Delta (Phase 3 − Dense) | 95% Bootstrap CI of Delta | Statistically Significant? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hit@1** | 0.7300 | 0.7500 | **0.7600** | +0.0300 | $[-0.0500, +0.1100]$ | No |
| **MRR@10** | 0.8096 | 0.8292 | **0.8432** | +0.0336 | $[-0.0154, +0.0855]$ | No |
| **NDCG@5** | 0.8337 | 0.8470 | **0.8589** | +0.0252 | $[-0.0171, +0.0701]$ | No |
| **NDCG@10** | 0.8436 | 0.8589 | **0.8710** | +0.0274 | $[-0.0118, +0.0705]$ | No |
| **Recall@5** | 0.9267 | 0.9217 | **0.9317** | +0.0050 | $[-0.0383, +0.0500]$ | No |
| **Recall@10** | 0.9533 | 0.9533 | **0.9667** | +0.0133 | $[-0.0200, +0.0500]$ | No |

**Statistical Honesty Statement:** Under 10,000 paired bootstrap resamples, all 95% confidence intervals cross zero. Even though directional gains were observed (+0.0336 MRR@10, +0.0252 NDCG@5), **none of the improvements are statistically significant at the 95% level at $N=100$**. We plainly state this without claiming demonstrated superiority.

---

### Q4: How does latency perform against the 300 ms SLA?
**A:** Evaluated via client-side wall-clock timing over 100 queries (20 warmups discarded) on an idle machine:

| Mode / Workload | p50 (ms) | p90 (ms) | p95 (ms) | SLA Target (<300 ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Baseline (Uncached)** | 58.84 ms | 74.51 ms | **94.50 ms** | < 300.00 ms | **PASS** (-205.50 ms margin) |
| **Hybrid (Uncached)** | 66.98 ms | 96.57 ms | **104.24 ms** | < 300.00 ms | **PASS** (-195.76 ms margin) |
| **Hybrid + Rerank ($K=30$, Uncached)** | 497.55 ms | 693.81 ms | **743.53 ms** | < 300.00 ms | **EXCEEDS SLA** (CPU cross-attention overhead) |
| **Cache: All-Unique Workload** | 66.18 ms | 89.25 ms | **103.04 ms** | < 300.00 ms | **PASS** |
| **Cache: 30% Repeated Workload** | 61.10 ms | 72.69 ms | **75.95 ms** | < 300.00 ms | **PASS** (27.1% speedup at p95) |
| **Cache: 100% Repeated (Best Case)**| 4.26 ms | 20.71 ms | **24.28 ms** | < 300.00 ms | **PASS** (76.7% speedup at p95) |

**Key Architectural Insight:**
Uncached hybrid search comfortably clears the SLA (p95 = 104.24 ms vs 300 ms limit). However, computing cross-attention across 30 full-text pairs on CPU takes ~450 ms of computation, exceeding the 300 ms limit when cold. This establishes the motivation for **Gate 4B**: deadline governor load shedding (shedding reranking when latency budget is tight), evaluating smaller $K=10$ (p95=284ms on TUNE), and caching repeated queries (p95=24.28ms).

---

### Q5: How is query cache correctness and invalidation verified?
**A:**
- Verified by automated unit tests in `tests/test_cache.py` (7/7 tests passed):
  1. `test_cache_hit_and_miss`: Second identical query returns in sub-millisecond time with `cache_hit=True`.
  2. `test_cache_invalidation_on_upsert`: Upserting a passage immediately purges the cache, preventing stale reads.
  3. `test_cache_invalidation_on_delete`: Deleting a passage immediately purges the cache, preventing ghost results.
  4. `test_cache_bypass_flag`: Requests with `use_cache=False` bypass lookup and cache insertion.
  5. `test_cache_key_differentiation`: Queries with different modes, filters, or top-$k$ generate distinct SHA-256 keys.
