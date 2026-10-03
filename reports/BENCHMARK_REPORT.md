# PRISMX Benchmarking & Evaluation Report

**System Name:** PRISMX Vector Database and Hybrid RAG Retrieval Engine  
**Author/Team:** Backend Engineering  
**Date:** 2026-10-03  
**Corpus:** MS MARCO 100,000 Passage Collection (Corpus A)  
**Hardware Platform:** Windows x86_64, 12 CPU Threads, Zero Paid Cloud APIs  

---

## 1. Executive Summary

This report establishes the baseline evaluation for **Phase 1 (Dense Semantic RAG)** and tracks system-wide compliance against all functional and non-functional requirements specified in the project charter.

### Headline Accomplishments:
1. **Full Scale Indexing:** Successfully indexed 100,000 canonical passages into a decoupled architecture combining **Qdrant v1.19.1** (dense HNSW vectors + dynamic BM25 sparse vectors with IDF modifiers) and a **SQLite Decoupled Text Store**.
2. **Ingestion Speed:** Completed in **0.992 hours** (3,572 seconds), beating the strict 2.0-hour requirement with >50% time budget margin.
3. **Latency Compliance (NFR-3):** Measured across 100 consecutive queries using client-side wall clock timing (D1 protocol). The system achieved a **p95 latency of 69.70 ms**, substantially outperforming the 300 ms target and the 250 ms internal safety margin.
4. **Information Retrieval Efficacy:** On the 500-query TUNE split, the dense retriever achieved **MRR@10 of 0.8749 [0.8506, 0.8991]** and **Recall@20 of 0.9750 [0.9610, 0.9880]**.
5. **RAGAS Baseline:** Established across 100 RAGAS evaluation queries with **Context Precision of 0.8035 [0.7347, 0.8662]** and **Context Recall of 0.9267 [0.8700, 0.9700]**.

---

## 2. Ingestion & Index Performance (Gate 2)

| Metric | Result | Target / Budget | Status |
| :--- | :--- | :--- | :--- |
| **Passages Indexed** | 100,000 | 100,000 min | **PASS** |
| **Total Ingestion Time** | 3,571.99 s (0.9922 hrs) | < 2.00 hours | **PASS** |
| **Dense Encoding Speed** | 28.8 passages/sec | Real CPU throughput | **PASS** |
| **Qdrant Upsert Speed** | 2,087.3 points/sec | High-throughput gRPC | **PASS** |
| **SQLite Ingest Speed** | 42,016 rows/sec | Chunked WAL transactions | **PASS** |
| **Hash Collisions** | 1 collision in 124,848 unique tokens | Minimal collision | **PASS** |
| **Label Leakage Check** | 0.00% (No ground truth labels stored) | Zero tolerance | **PASS** |

---

## 3. Phase 1 Dense Retrieval Evaluation (Gate 3)

### 3.1 TUNE Split Metrics (N = 500 queries, 10,000 Bootstrap Resamples)

Evaluated sequentially against the full 100,000 passage index at candidate depth 20:

| Metric | Mean Score | 95% Confidence Interval |
| :--- | :--- | :--- |
| **Hit@1** | 0.8200 | [0.7860, 0.8520] |
| **Success@5** | 0.9500 | [0.9300, 0.9680] |
| **Success@10** | 0.9640 | [0.9460, 0.9800] |
| **Recall@5** | 0.9443 | [0.9240, 0.9630] |
| **Recall@10** | 0.9580 | [0.9400, 0.9740] |
| **Recall@20** | 0.9750 | [0.9610, 0.9880] |
| **MRR@10** | 0.8749 | [0.8506, 0.8991] |
| **NDCG@5** | 0.8873 | [0.8649, 0.9093] |
| **NDCG@10** | 0.8919 | [0.8706, 0.9128] |
| **Dup-Aware Hit@1 (PATCH-4)** | 0.8200 | [0.7860, 0.8520] |
| **Dup-Aware Recall@5 (PATCH-4)**| 0.9443 | [0.9240, 0.9630] |

*Throughput:* 21.41 queries per second (23.36s total evaluation time).

---

## 4. Query Latency Benchmark (NFR-3 Protocol D1/D2/D3)

Measured over 20 warm-up queries (excluded from percentiles) followed by **100 consecutive distinct BENCH queries** executed strictly sequentially over HTTP against `POST /search`. Percentiles computed via numpy linear interpolation.

| Thread Configuration | Warmup Mean | p50 (Median) | p90 | p95 (Target <300ms) | p99 | Max | NFR-3 Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Default Threads (12T)** | 60.86 ms | 50.40 ms | 65.48 ms | **69.70 ms** | 77.66 ms | 123.40 ms | **PASS** (<250ms target) |
| **Constrained Threads (4T)** | 53.42 ms | 49.76 ms | 60.97 ms | **66.32 ms** | 91.00 ms | 95.96 ms | **PASS** (<250ms target) |

### Per-Component Latency Breakdown (Mean ms):
- **Query Dense Encoding:** 29.6 ms
- **Qdrant HNSW Vector Search:** 9.1 ms
- **SQLite Decoupled Text Fetch:** 5.0 ms
- **Network / Serialization Overhead:** ~6.7 ms
- **Total End-to-End Latency:** ~50.4 ms

---

## 5. RAGAS Evaluation Baseline (Family A Non-LLM)

Evaluated on 100 dedicated RAGAS evaluation queries (`split_ragas.json`) with 10,000 bootstrap resamples:

| Metric | Phase 1 Dense Baseline | 95% Confidence Interval | Phase 2 Target |
| :--- | :--- | :--- | :--- |
| **Context Precision** | **0.8035** | [0.7347, 0.8662] | > 0.75 (NFR-1 Met) |
| **Context Recall** | **0.9267** | [0.8700, 0.9700] | > 0.70 (NFR-2 Met) |

*Artifact Reference:* Stored in `results/phase1/ragas_eval.json` and `results/phase1/metrics.json`.

---

## 6. Architecture & Contract Compliance

1. **Decoupled Architecture (PATCH-2 / ADR-007):**
   - High memory footprint eliminated by storing passage text strictly in SQLite.
   - Vector database payloads contain only indexed metadata (`passage_id`, `category`, `source`).
   - Order preservation and corruption tolerance verified via unit test suite `tests/test_service.py` (2/2 passing).
2. **Dynamic IDF Modifier (PATCH-1 / ADR-006):**
   - Qdrant sparse vectors use `models.Modifier.IDF`, verified to adapt dynamically to collection term statistics without requiring full reindexing.
   - `avgdl_ref` frozen at index build time (33.6145) with real-time drift tracking exposed in `GET /meta`.
3. **Deterministic Testing:**
   - 25 total unit and integration tests passing (`tests/test_gate0.py`, `test_gate1.py`, `test_gate2.py`, `test_filters.py`, `test_fusion.py`, `test_service.py`).
