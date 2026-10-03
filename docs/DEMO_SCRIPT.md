# PRISMX: Live Evaluation & Demonstration Script

This script provides an exact, step-by-step walkthrough for hackathon judges and evaluators to test all functional, non-functional, and architectural requirements of PRISM-X.

---

## 1. Seeded Demo Query Provenance Matrix (Seed 42) — [Illustration, N=6, Not Evidence]

The 6 example chips displayed in the search interface are drawn deterministically from `data/c100k_raw/tune_raw_500.json` (**Random Seed: 42**).
- **Status:** **Illustration, N=6, not evidence.** Formal empirical claims are established strictly on the 100-query BENCH set (`results/c100k_raw/bench_eval_results.json`).
- **Draw Code:**
  ```python
  import json, random
  with open("data/c100k_raw/tune_raw_500.json") as f:
      tune = json.load(f)
  rng = random.Random(42)
  sample = rng.sample(tune, 6)
  ```
- **Selected QIDs:** `[304343, 1096404, 637720, 970924, 501994, 716132]`
- **Raw API Responses:** Persisted in [`results/demo/seed42_chip_responses.json`](file:///c:/Users/manis/OneDrive/Desktop/main_adrosonic/results/demo/seed42_chip_responses.json)

| # | Demo Query | QID | Gold PID | Dense Top-1 (Class) | Hybrid Top-1 (Class) | PRISM-X Top-1 (Class) | Cross-Encoder Effect |
| :-: | :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **1** | `how much did rogue one make at the box office` | 304343 | `raw_9916` | `raw_9916` *(exact gold)* | `raw_9919` *(sibling)* | **`raw_9916` *(exact gold)*** | **Recovers exact gold over hybrid sibling** |
| **2** | `how much can you file for in small claims court?` | 1096404 | `raw_82491` | `raw_82499` *(sibling)* | `raw_82499` *(sibling)* | `raw_82499` *(sibling)* | Sibling contains statutory claim limits *(manually read)* |
| **3** | `what does flourish mean` | 637720 | `raw_53267` | `raw_53268` *(sibling)* | `raw_53268` *(sibling)* | `raw_53268` *(sibling)* | Sibling dictionary definition candidate *(manually read)* |
| **4** | `where do i find nutritional yeast` | 970924 | `raw_53226` | `raw_53226` *(exact gold)* | `raw_53226` *(exact gold)* | **`raw_53226` *(exact gold)*** | Unanimous agreement on grocery section |
| **5** | `square footage price for electrical` | 501994 | `raw_53792` | `raw_53793` *(sibling)* | `raw_53793` *(sibling)* | **`raw_53792` *(exact gold)*** | **Elevates gold pricing breakdown over sibling (+1 rank)** *(sibling manually read)* |
| **6** | `what is an mda` | 716132 | `raw_27545` | `raw_27554` *(sibling)* | `raw_27554` *(sibling)* | **`raw_27545` *(exact gold)*** | **Elevates medical definition gold over sibling (+1 rank)** *(sibling manually read)* |

> **Audit & Explanation of Previous Discrepancy:**
> In Gate 5.8, an earlier draft presented a manually filtered set of queries (stuffed flounder, cardamom, Skyrim, etc.) and erroneously hypothesized 6/6 exact-gold hits with identical top-1 across all modes. That did not match real retrieval behavior (where dense BENCH Hit@1 is 0.42 and hybrid is 0.41, with 48–51% of top-1 results being unjudged sibling passages). 
> The table above reflects the true, unadulterated `random.Random(42).sample(tune, 6)` execution queried against the live running API:
> - **Dense Hit@1:** 2/6 = 33.3% exact gold (matches BENCH ~42%)
> - **Hybrid Hit@1:** 1/6 = 16.7% exact gold (5/6 siblings)
> - **PRISM-X Hit@1:** 4/6 = 66.7% exact gold (+50.0% gain over hybrid, promoting gold for Chip 1, 5, and 6)
> Sibling passages come from the same originating search session in MS MARCO; annotators labeled one gold passage and stopped, leaving sibling passages unjudged. The Cross-Encoder effectively discerns the specific human-labeled gold passage over near-duplicate siblings.

---

## 2. 7-Step Judge Evaluation Walkthrough

### Step 1: Ingestion & Collection Health Verification
- **Target URL:** `http://127.0.0.1:5173/` or API `http://127.0.0.1:8000/meta`
- **Action:** Inspect the top header status pills and collection metadata.
- **Expected Result:**
  - `● Live` status badge active.
  - `100K MS MARCO` index pill showing 100,008 passages indexed.
  - Config hash: `8e1000d5...`.
  - True avgdl: `53.2501` with zero drift.
- **Data File Shown:** `data/c100k_raw/manifest.json`.

---

### Step 2: Dense Baseline Search (Phase 1)
- **Target URL:** `http://127.0.0.1:5173/`
- **Query:** Click chip: `how much did rogue one make at the box office`
- **Action:** Select mode **Dense Only** (`🧠`).
- **Expected Result:**
  - Results returned with **Similarity** score (e.g. `0.9049`).
  - Score displayed in muted pill with tooltip: *"Model score (uncalibrated logit or similarity), not a probability."*
  - Latency breakdown shows `Encode` (~40ms) and `Dense` (~10ms).
- **Data File Shown:** `results/c100k_raw/raw_latency_dense_bench100.csv`.

---

### Step 3: Hybrid Search with Documented Fusion (Serving Default)
- **Target URL:** `http://127.0.0.1:5173/`
- **Query:** Same query (`how much did rogue one make at the box office`)
- **Action:** Switch mode to **Hybrid (Serving Default)** (`⚖️`).
- **Expected Result:**
  - Dual-vector retrieval combining BGE-small dense and BM25 sparse IDF.
  - Linear min-max weighted fusion with $\alpha=0.80$.
  - Score labeled **Fused score** (range $[0, 1]$).
  - SLA pill displays: `Hybrid SLA <300ms PASS (idle p95=89.02ms)`.
- **Data File Shown:** `results/phase2/fusion_tuning_tune.json`.

---

### Step 4: PRISM-X Cross-Encoder Reranking & 3-Column Comparison
- **Target URL:** `http://127.0.0.1:5173/comparison`
- **Query:** `square footage price for electrical` (Chip 5)
- **Action:** Inspect side-by-side columns: Dense vs Hybrid vs PRISM-X.
- **Expected Result:**
  - Dense and Hybrid place sibling `raw_53793` at Top-1.
  - PRISM-X cross-encoder reorders candidates and elevates the verified **Gold passage `raw_53792`** to Top-1 with badge `★ Gold`.
  - Rank movement badge shows `▲ +1`.
  - Score labeled **Rerank score** (uncalibrated logit, e.g. `2.8216`).
- **Data File Shown:** `results/c100k_raw/bench_eval_results.json` and `results/demo/seed42_chip_responses.json`.

---

### Step 5: Pre-Retrieval Metadata Filtering
- **Target URL:** `http://127.0.0.1:5173/`
- **Query:** `square footage price for electrical`
- **Action:** Select Category filter dropdown: **`LOCATION`**.
- **Expected Result:**
  - 100% of returned passages belong to category `LOCATION` (e.g. `raw_43954`, `raw_16538`).
  - Zero out-of-filter results (0 false positives).
  - Filter applied pill displayed: `Filter applied`.
- **Data File Shown:** `scratch/audit_fr4_filter.py` test logs (100% exact filter overlap).

---

### Step 6: Live Document Upsert/Delete with Stale Read Prevention
- **Target URL:** Terminal / Test Runner
- **Command:** `pytest tests/test_gate5_comprehensive.py -k "upsert or delete" -v`
- **Expected Result:**
  - Upsert of new test document is immediately retrievable via `POST /search` on the very next query.
  - Immediate subsequent delete removes document from both Qdrant and SQLite.
  - Query cache is synchronously invalidated on both operations, preventing stale reads.
  - Both tests PASS (100% compliance with FR-5 and NFR-5).
- **Data File Shown:** `tests/test_gate5_comprehensive.py`.

---

### Step 7: Architecture Walkthrough & Evaluation Dashboard
- **Target URL:** `http://127.0.0.1:5173/architecture` and `http://127.0.0.1:5173/evaluation`
- **Action:** Review system architecture diagrams and statistical confidence intervals.
- **Expected Result:**
  - Decoupled storage diagram (Qdrant vectors + SQLite WAL text).
  - Statistically verified retrieval metrics: MRR@10 paired gain $+0.0583$ (95% CI $[+0.0015, +0.1165]$ excludes 0).
  - Strict SLA separation: Challenge SLA (300 ms) vs Internal Target Ceiling (250 ms).
  - Transparent governor explanation of non-idle tail latency spikes.
- **Data File Shown:** `results/c100k_raw/latency_benchmark.json` and `results/c100k_raw/bench_eval_results.json`.

---

## 3. Fallback Strategies (Network / Groq Offline)

1. **Groq API Rate Limit or Outage:**  
   If the Groq API returns HTTP 429 or is unreachable, the system automatically falls back to extractive answer synthesis directly from the Top-1 retrieved passage text with explicit citation indexing, ensuring uninterrupted judge evaluation.
2. **Offline Local Serving Mode:**  
   The frontend includes a pre-recorded mock client toggle (`recordedMode`) that can replay verified benchmark traces offline if the local backend server is stopped.
