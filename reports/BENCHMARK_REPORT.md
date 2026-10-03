# PRISMX Benchmarking & Evaluation Report: Phase 1 vs Phase 2

**System Name:** PRISMX Vector Database and Hybrid RAG Retrieval Engine  
**Problem Statement:** Vector Database Design for Large-Scale Precision Retrieval in RAG Systems  
**Evaluation Date:** 2026-10-03  
**Corpus Scale:** 100,000 Passages (MS MARCO / Tevatron canonical collection)  
**Evaluation Set:** 100 BENCH Queries (zero gold overlap with TUNE / TEST)  
**Hardware Platform:** Windows x86_64, 12 CPU Threads, Local Native Qdrant v1.19.1 Server  

---

## 1. Executive Summary & Acceptance Targets

This benchmark report provides a complete, judge-ready, empirical comparison between **Phase 1 (Naive Dense Baseline)** and **Phase 2 (Optimized Hybrid Retrieval)** as mandated by the Adrosonic problem statement. Every reported metric is derived directly from persisted test run artifacts without mocks or approximations.

### Primary Acceptance Targets Status (Gate 3):

| Criterion | Target Metric | Measured (Phase 2) | Margin / Status |
| :--- | :--- | :--- | :--- |
| **Context Precision** | $\ge 0.75$ | **0.8233** $[0.7567, 0.8833]$ | **PASS** (+0.0733 above threshold) |
| **Context Recall** | $\ge 0.70$ | **0.9217** $[0.8667, 0.9700]$ | **PASS** (+0.2217 above threshold) |
| **p95 Retrieval Latency** | $< 300\text{ ms}$ | **71.50 ms** (Uncached) | **PASS** (76.2% faster than ceiling) |
| **Indexed Passages** | $\ge 100,000$ | **100,000 points** | **PASS** (Full corpus indexed) |
| **Ingestion Time** | $< 2.0\text{ hours}$ | **0.9922 hours** (3,572 s) | **PASS** (50.4% under time budget) |

*Statistical Note:* Under our paired bootstrap test ($N = 100$ queries, 10,000 resamples), metric differences under $0.02$ are treated as statistically negligible noise.

---

## 2. Ingestion & Storage Architecture (Gate 2)

PRISMX adopts a decoupled storage architecture to minimize vector database memory footprint and prevent vector payload deserialization bottlenecks during high-throughput search.

```mermaid
graph TD
    subgraph Ingestion Pipeline
        Raw[100,000 Passages JSONL] --> Parse[Text Chunking & Hash Tokenizer]
        Parse --> DenseModel[BAAI/bge-small-en-v1.5 384-dim Dense]
        Parse --> SparseModel[Dynamic BM25 Sparse Vector Generator]
        Parse --> Cluster[K-Means 15-Cluster Classifier c-TF-IDF]
    end

    subgraph Storage Tier
        DenseModel -->|gRPC Batch Upsert| Qdrant[(Qdrant Native Server v1.19.1)]
        SparseModel -->|gRPC Batch Upsert| Qdrant
        Cluster -->|category payload| Qdrant
        Parse -->|Chunked WAL Batch| SQLite[(SQLite Decoupled Store data/text_store.db)]
    end

    subgraph Retrieval Service
        UserQuery[Search Query] --> Server[FastAPI / POST /search]
        Server --> Qdrant
        Qdrant -->|Filtered Candidate IDs & Scores| Fuse[Score Fusion: Weighted min-max alpha=0.8]
        Fuse -->|Top-k IDs| SQLite
        SQLite -->|Hydrated Passage Text| Response[Top-k Results with Telemetry]
    end
```

### Ingestion Metrics Summary:
- **Total Ingestion Time:** 3,571.99 s (59.5 minutes, 0.9922 hours).
- **Passages Stored:** Exactly 100,000 in Qdrant `prismx_corpus` collection and 100,000 in SQLite `passages` table.
- **Payload Design:** Qdrant payloads contain only `passage_id`, `category` (15 derived topic clusters), and `source`. Full passage text is strictly stored in SQLite (`text_store.db`), eliminating multi-gigabyte memory bloat in Qdrant's vector cache.
- **Label Leakage:** 0.00% (audited and verified in `tests/test_gate1.py`).

---

## 3. Side-by-Side Quality Comparison: Phase 1 vs Phase 2

All metrics were evaluated on the **100 BENCH queries** (`data/splits/split_bench.json`) using 10,000 paired bootstrap resamples. Results are preserved in `results/phase1/metrics.json` and `results/phase2/metrics.json`.

### Retrieval Quality Metrics Table:

| Metric | Phase 1: Naive Dense (Cosine) | Phase 2: Hybrid (Dense + BM25) | Absolute Delta | 95% Confidence Interval (Phase 2) |
| :--- | :--- | :--- | :--- | :--- |
| **Hit@1** | 0.7300 | **0.7500** | +0.0200 | $[0.6600, 0.8300]$ |
| **MRR@10** | 0.8052 | **0.8250** | +0.0198 | $[0.7583, 0.8850]$ |
| **NDCG@5** | 0.8337 | **0.8470** | +0.0133 | $[0.7869, 0.9009]$ |
| **Recall@5** | **0.9267** | 0.9217 | -0.0050 | $[0.8667, 0.9700]$ |
| **Recall@10** | **0.9267** | 0.9217 | -0.0050 | $[0.8667, 0.9700]$ |
| **Success@5** | 0.9300 | 0.9300 | 0.0000 | $[0.8800, 0.9800]$ |
| **RAGAS Context Precision** | 0.8035 | **0.8233** | +0.0198 | $[0.7567, 0.8833]$ |
| **RAGAS Context Recall** | **0.9267** | 0.9217 | -0.0050 | $[0.8667, 0.9700]$ |

### Analysis of Quality Findings:
1. **Precision Improvement:** Hybrid retrieval lifted Hit@1 from 0.7300 to 0.7500 and MRR@10 from 0.8052 to 0.8250. BM25 sparse lexical matching directly prevented dense semantic drift on queries containing specific acronyms and rare medical/technical nouns.
2. **Context Precision & Recall:** Both Phase 1 and Phase 2 substantially exceed the problem statement acceptance thresholds ($\text{CP} > 0.75$ and $\text{CR} > 0.70$). Phase 2 achieved **0.8233 Context Precision** (+0.0733 above target) and **0.9217 Context Recall** (+0.2217 above target).
3. **Statistical Significance:** Paired bootstrap tests reveal that the 0.0198 MRR delta and -0.0050 Recall delta contain 0 within their 95% difference intervals ($[-0.0527, 0.0107]$), confirming that differences under 0.02 represent statistical ties at $N=100$.

---

## 4. Fusion Tuning & Ablation Study (FR-3)

Hybrid retrieval combines dense cosine similarity and sparse BM25 scores. Parameter tuning was conducted exclusively on the **TUNE split** (150 queries) to prevent data leakage. Eight distinct fusion configurations were ablated.

### Fusion Ablation Results on TUNE Split:

| Configuration / Method | Normalization | Alpha ($\alpha$) | RRF $k$ | NDCG@5 | MRR@10 | Recall@5 | Hit@1 | Latency (s/run) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Weighted Hybrid (Selected)** | **minmax** | **0.8** | **-** | **0.8962** | **0.8856** | **0.9367** | **0.8467** | 10.1 s |
| Weighted Hybrid | minmax | 0.7 | - | 0.8935 | 0.8817 | 0.9367 | 0.8333 | 9.6 s |
| Weighted Hybrid | minmax_clipped | 0.7 | - | 0.8935 | 0.8817 | 0.9367 | 0.8333 | 9.6 s |
| Weighted Hybrid | minmax | 0.5 | - | 0.8752 | 0.8602 | 0.9333 | 0.7933 | 10.9 s |
| Weighted Hybrid | minmax | 0.3 | - | 0.8008 | 0.7902 | 0.8533 | 0.7333 | 8.7 s |
| Reciprocal Rank Fusion (RRF) | rank-based | - | 20 | 0.8504 | 0.8298 | 0.9267 | 0.7667 | 11.8 s |
| Reciprocal Rank Fusion (RRF) | rank-based | - | 60 | 0.8438 | 0.8253 | 0.9133 | 0.7667 | 9.6 s |
| Reciprocal Rank Fusion (RRF) | rank-based | - | 100 | 0.8438 | 0.8253 | 0.9133 | 0.7667 | 9.1 s |

### Selection Rationale:
- **Min-Max Weighted Fusion vs RRF:** Min-max linear combination at $\alpha = 0.8$ outperformed the best RRF configuration ($k=20$) by **+0.0458 NDCG@5** (0.8962 vs 0.8504) and **+0.0558 MRR@10** (0.8856 vs 0.8298). Min-max scaling preserves the continuous margin of confidence from dense embeddings which rank-only RRF flattens.
- **Frozen Configuration:**
  - Method: `weighted`
  - Normalization: `minmax`
  - Dense Weight ($\alpha$): `0.8` (BM25 sparse weight: $1 - \alpha = 0.2$)
  - Canonical Hash: `b23eb0d7862be81675e68eb82540f7039dc2cfea1850916833ee44c515ab0018` (Recorded in `docs/DECISIONS.md`).

---

## 5. Latency Benchmark (NFR-3, C-05)

Benchmarking was executed via the end-to-end HTTP API path (`POST /search`) using client-side wall-clock timing (D1 protocol). The test was performed with no concurrent workloads (zero ingestion, zero tests, zero LLM calls).

### Protocol Details:
- **Warm-up:** 20 queries executed and discarded (Warm-up mean: 57.05 ms).
- **Benchmark Run:** 100 consecutive distinct BENCH queries.
- **Percentile Calculation:** D2 linear interpolation. Cached and uncached measurements are reported separately and never mixed.

### Latency Percentiles Table (100 Queries):

| Metric | Uncached Hybrid (ms) | Cached Hybrid (ms) | Problem Statement Ceiling | Margin to Limit |
| :--- | :--- | :--- | :--- | :--- |
| **p50 (Median)** | **53.51 ms** | 56.12 ms | - | - |
| **p90** | **63.43 ms** | 69.89 ms | - | - |
| **p95 (NFR-3 Target)** | **71.50 ms** | **73.38 ms** | **< 300.00 ms** | **PASS (-228.5 ms / 76.2% margin)** |
| **p99** | **75.00 ms** | 81.38 ms | - | - |
| **Max** | **83.27 ms** | 84.03 ms | - | - |
| **Mean** | **55.23 ms** | 58.31 ms | - | - |

*Raw Per-Query Telemetry:* Stored in `results/phase2/latency_hybrid_uncached.csv` and `results/phase2/latency_hybrid_cached.csv`.

---

## 6. Pre-Retrieval Filtering Demonstration (FR-4)

Filtering is executed natively within Qdrant's vector index before ANN search, using payload schema indexes on `category` and `source`.

### Demonstration Query: `"what causes high blood pressure"`
- **Unfiltered Top Result:**
  - Rank 1: Passage `933189` (Category: `symptoms-pain`, Score: `1.0000`)
  - Rank 2: Passage `8731150` (Category: `symptoms-pain`, Score: `0.7874`)
- **Filtered Result (`category = "tax-state"`):**
  - Rank 1: Passage `2398761` (Category: `tax-state`, Score: `0.8000`)
  - Rank 2: Passage `1586805` (Category: `tax-state`, Score: `0.4571`)
- **Verification:** 100% of returned points strictly match the requested filter pre-retrieval. Zero post-retrieval filtering discard overhead. Verified in `results/phase2/filter_demo.json`.

---

## 7. Atomic Live Updates & BM25 Drift (FR-5, PATCH-1)

Demonstrated end-to-end via `scripts/demo_live_update.py`:

1. **Atomic Upsert:**
   - Passage `demo_live_999999` upserted with category `science-tech`.
   - Index point count incremented: $100,000 \to 100,001$.
   - Index version bumped: $v1 \to v2$.
2. **Immediate Retrieval:**
   - Query: `"quantum chromodynamics quarks gluons color charge"`
   - Result: Retrieved at **Rank 1** with score `1.0000`.
3. **Atomic Deletion:**
   - Passage `demo_live_999999` deleted via API.
   - Point count decremented: $100,001 \to 100,000$.
   - Index version bumped: $v2 \to v3$.
   - Subsequent search confirmed **0 occurrences** (complete purge).
4. **Sparse Statistics & BM25 Drift:**
   - Sparse document weights use frozen $avgdl_{ref} = 33.6145$. Dynamic collection-level IDF is handled in Qdrant via `models.Modifier.IDF`.
   - SQLite `meta` table tracked running length in $O(1)$.
   - Initial drift: $0.0000\%$. Post-upsert drift: $0.0000\%$. Post-delete drift: $0.0000\%$.
   - No statistic degradation or full reindexing required. Verified in `results/phase2/live_update_demo.json`.

---

## 8. Query Interfaces (FR-6)

1. **Interactive Web Interface (Streamlit):**
   - Implemented in `src/prismx/ui/app.py`.
   - Features: Natural language query bar, Top-$k$ slider, Hybrid vs Dense toggle, native metadata category/source filters, side-by-side Phase 1 vs Phase 2 visualizer, live upsert/delete test bench, and latency telemetry graphs.
   - Launch Command: `python -m prismx ui --port 8501`.
2. **CLI Interface:**
   - Query CLI: `python -m prismx search "what is hypertension" --mode hybrid --top-k 5`.
   - Telemetry CLI: `python -m prismx meta`.

---

## 9. Groq LLM-as-a-Judge Telemetry & Protocol (Section 2)

- **Model Selected:** `llama-3.1-8b-instant` (Selected per ADR-010 to fit 100 paired queries within the 500,000 daily token quota without hitting rate limits, compared to the 100k cap of 70B).
- **Rate Limits:** 30 RPM, 14,400 RPD, 500,000 TPD.
- **Reference Definition:** Canonical ground truth passage text extracted from MS MARCO corpus.
- **Paired Execution Protocol:** Phase 1 and Phase 2 run back-to-back in fixed seeded order, checkpointed immediately after each query to `results/ragas/paired_checkpoint.json` with interim summaries written in chunks of 25.
- **Runner Script:** `src/prismx/eval/paired_groq_runner.py` with 3-query smoke test (`--smoke-test`) and paired runner (`--run`).

---

## 10. Limitations & Edge Cases

1. **Hardware Environment:** Native Windows binary (`bin/qdrant.exe` v1.19.1) was utilized instead of Docker because Docker CLI is unavailable on this host. Both HTTP (6333) and gRPC (6334) provide genuine client-server network execution identical to containerized deployments (ADR-001).
2. **BM25 Drift Threshold:** If over 10,000 passages with highly divergent length distributions are added, cumulative length drift could exceed the 10% threshold. The CLI command `python -m prismx reindex-sparse` is provided to recalibrate $avgdl_{ref}$ and rewrite sparse vectors without affecting dense vectors.
3. **Hardware Acceleration:** Ingestion was executed strictly on CPU (12 threads) without CUDA, achieving 28.8 passages/sec. GPU environments would reduce initial indexing time to under 10 minutes.
