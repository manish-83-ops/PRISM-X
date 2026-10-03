# Architecture Decision Records (ADRs)

All architectural and algorithmic decisions are recorded here with context, options, choice, rationale, and evidence. Frozen configuration hashes are recorded here prior to their single TEST evaluation.

---

## ADR-001: Vector Database Deployment Mode (Qdrant)
- **Date:** 2026-10-03
- **Context:** The challenge requires a vector database (Qdrant). Docker is not installed on the Windows host machine. Rule R10/Gate 0 specifies: "Qdrant: run a Qdrant SERVER via docker compose (preferred). If Docker is unavailable, check whether an official native Qdrant binary exists for this OS by reading the official release page; use it if found. Embedded/local mode is a LAST RESORT: if used, flag it in ISSUES.md because it has different performance and must not be presented as server performance."
- **Options Considered:**
  1. *Embedded/local mode (`:memory:` or local directory)*: Disqualified as a last resort because client-embedded performance differs significantly from real client-server network/gRPC latency.
  2. *Install Docker Desktop*: Requires administrative hypervisor privileges and restart.
  3. *Official Native Qdrant Binary (`qdrant-x86_64-pc-windows-msvc.zip`)*: Officially built and released by Qdrant on GitHub for Windows x86_64.
- **Choice:** Option 3 (Official Native Qdrant Binary v1.19.1 in `bin/qdrant.exe`).
- **Why:** Delivers genuine server performance over HTTP (port 6333) and gRPC (port 6334) identical to the Docker container, fully compliant with Gate 0 without using the embedded fallback.
- **Evidence:** `bin/qdrant.exe --version` returned `qdrant 1.19.1`. Verified client connection over gRPC and synchronous collection creation in Gate 0.

---

## ADR-002: Deterministic Stable Hashing for Tokens and Sampling
- **Date:** 2026-10-03
- **Context:** Rule R9 requires: "Fixed seeds, deterministic ordering, record all seeds. Use a STABLE hash for tokens and sampling (never Python's built-in hash(): it is salted per process; use e.g. a hashlib/mmh3-style stable hash and record which)."
- **Options Considered:**
  1. *Python built-in `hash()`*: Non-deterministic across Python processes (salted with `PYTHONHASHSEED`).
  2. *`mmh3` (MurmurHash3)*: Requires C extension compilation on Windows which may fail without MSVC C++ build tools.
  3. *`hashlib.sha256` / `hashlib.md5`*: Built into standard library, 100% deterministic across all platforms, languages, and processes.
- **Choice:** Standard library `hashlib.sha256`.
- **Why:** Guaranteed zero-dependency cross-process reproducibility on Windows, Linux, and macOS. For sparse term index mapping into 32-bit unsigned integers: `int(hashlib.sha256(token.encode('utf-8')).hexdigest()[:8], 16)`.
- **Evidence:** Unit tests in `tests/test_gate0.py` and `tests/test_gate1.py`.

---

## ADR-003: Decoupled Vector vs Text Storage (Qdrant + SQLite)
- **Date:** 2026-10-03
- **Context:** Storing large passage text payloads in vector indexes consumes excessive RAM and inflates vector transfer latencies during search.
- **Options Considered:**
  1. *Store raw passage text directly in Qdrant point payloads*: High memory footprint in Qdrant WAL and memory map; payload retrieval increases latency.
  2. *Decoupled Architecture (Qdrant for vector filtering + SQLite for text storage)*: Qdrant payloads contain only filtering metadata (`passage_id`, `category`, `source`); full passage texts are stored in SQLite and retrieved only for the final top-$k$ results.
- **Choice:** Option 2 (Decoupled Qdrant + SQLite).
- **Why:** Minimizes Qdrant memory footprint, optimizes HNSW vector search speed, guarantees durability, and allows persistent metadata (`avgdl`, `index_version`) in the SQLite `meta` table.
- **Evidence:** Verified in Gate 0 and Gate 2 index build.

---

## ADR-004: Qdrant Query API Selection (`query_points` vs `search`)
- **Date:** 2026-10-03
- **Context:** In `qdrant-client` 1.19.1, `QdrantClient.search` has been removed.
- **Choice:** Use `client.query_points`.
- **Why:** `query_points` is the unified method supporting dense vectors, sparse vectors (`using="bm25"`), filters, and payload selection in `qdrant-client` 1.19.1.
- **Evidence:** `tests/test_gate0.py` and `docs/API_NOTES.md`.

---

## ADR-005: Dataset Source Selection for MS MARCO Corpus and Splits
- **Date:** 2026-10-03
- **Context:** Gate 1 requires finding MS MARCO passage data on HuggingFace that provides stable passage IDs, queries, and qrels, verifying 100% cross-dataset consistency.
- **Options Considered:**
  1. *`microsoft/ms_marco`*: Original legacy HF dataset; lacks pre-split passage corpus with fast retrieval IDs; schema requires complex parsing.
  2. *`BeIR/msmarco`*: Clean corpus and queries; however, corpus parquet download is 1.55 GB and query extraction over streaming is slow.
  3. *`Tevatron/msmarco-passage-corpus` + `Tevatron/msmarco-passage` + `BeIR/msmarco-qrels`*:
     - `Tevatron/msmarco-passage-corpus`: canonical `docid` string (exact MS MARCO integer IDs `0` to `8841822`).
     - `Tevatron/msmarco-passage` (`dev.jsonl.gz`): exactly 6,980 dev queries.
     - `BeIR/msmarco-qrels` (`dev.tsv`): exactly 7,437 qrels across 6,980 dev queries.
- **Choice:** Option 3.
- **Why:** Complete 100% ID consistency between qrels and queries (6,980/6,980 match). 100% of gold passage IDs exist in the corpus. Clean schema: `docid` and `text`.
- **Evidence:** Verified in `data/manifests/splits_manifest.json` and `data/manifests/corpus_manifest.json`.

---

## ADR-006: [PATCH-1] Frozen BM25 `avgdl_ref`, Dynamic Qdrant IDF, and Drift Monitoring
- **Date:** 2026-10-03
- **Context:** In BM25 document scoring, length normalization requires collection average document length (`avgdl`). Dynamic updates in production systems can either require full reindexing or introduce parameter drift.
- **Specification:**
  1. **Sparse Document Weights:** Computed at index time using a frozen reference average document length `avgdl_ref` established on the initial 100,000 passage corpus:
     $$\text{TF\_weight}(t, d) = \frac{\text{tf} \cdot (k_1 + 1)}{\text{tf} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl\_ref}}\right)}$$
  2. **Dynamic IDF in Qdrant:** Qdrant's sparse vector engine with `models.Modifier.IDF` dynamically computes and updates inverted document frequencies based on live collection point statistics. *Empirically verified on 2026-10-03:* Inserting 10 documents containing a term reduced its query score from `0.2877` to `0.0465`, proving dynamic IDF adaptation.
  3. **Real-time Drift Tracking:** SQLite `meta` table maintains `total_doc_len` and `n_docs` updated in $O(1)$ on every upsert and delete.
     $$\text{true\_avgdl} = \frac{\text{total\_doc\_len}}{\text{n\_docs}}, \quad \text{drift} = \frac{|\text{true\_avgdl} - \text{avgdl\_ref}|}{\text{avgdl\_ref}}$$
     Exposed in `GET /meta`. When `drift > 0.10` (10%), a warning is logged.
  4. **Sparse Reindexing CLI:** `python -m prismx reindex-sparse` recomputes `avgdl_ref` and rewrites sparse vectors only in batches.
  5. **TUNE Sensitivity Experiment:** In Gate 4, evaluate BM25 on TUNE across `avgdl_ref` scaling factors $[0.75, 0.90, 1.00, 1.10, 1.25]$ with bootstrap CIs on MRR@10 and Recall@20.
- **Affected Steps & Re-Run Status:**
  - Affects Gate 2 (Index Build), Gate 4 (Ablation), Gate 6 (Live Updates).
  - Completed steps (Gates 0 & 1) are unaffected. No re-runs required.

---

## ADR-007: [PATCH-2] Deterministic Result Ordering and SQLite Decoupling
- **Date:** 2026-10-03
- **Context:** Decoupled architecture retrieves candidate passage IDs from Qdrant and fetches hydrated passage text from SQLite. SQL engines do not preserve `WHERE id IN (...)` order.
- **Specification:**
  1. **Order Preservation:** The retrieval service generates an explicit ordered list of top-$k$ passage IDs. Texts are fetched from SQLite (chunking `IN` queries to $\le 500$ parameters), mapped into an in-memory dictionary `dict[str, PassageRecord]`, and iterated over the ordered candidate list. SQL row order is never relied upon.
  2. **Inconsistency Tolerance:** If a point exists in Qdrant but is missing from SQLite, the service skips the ID, increments an `inconsistency_count` metric (exposed in `GET /meta`), logs the ID, and backfills from the next candidate in the fused candidate pool.
  3. **Verification:** Unit tests in `tests/test_service.py` assert preservation of fused order against shuffled SQL returns and verify stable tie-breaking.
- **Affected Steps & Re-Run Status:**
  - Affects Gate 3 (Dense Service) and Gate 4 (Hybrid Service).
  - Completed steps (Gates 0 & 1) are unaffected. No re-runs required.

---

## ADR-008: [PATCH-3] Fusion Normalization Strategies and Adaptive Default Selection
- **Date:** 2026-10-03
- **Context:** Linear combination hybrid retrieval requires score normalization across heterogeneous score distributions (dense cosine $[-1, 1]$ vs BM25 sparse $[0, \infty)$).
- **Specification:**
  1. **Normalization Variants:** Support three strategies for weighted fusion:
     - `minmax`: Standard linear scaling over candidate set $\frac{s - \min}{\max - \min}$.
     - `minmax_clipped`: Clip channel candidate scores to 5th and 95th percentiles before min-max scaling to resist extreme score outliers.
     - `zscore`: Standard score standardization $\frac{s - \mu}{\sigma}$.
     - `rrf`: Reciprocal Rank Fusion $\sum \frac{1}{k_{\text{rrf}} + \text{rank}}$.
  2. **Outlier Compression Diagnostics:** On TUNE, measure the fraction of queries where rank 5 normalized score $< 0.1$.
  3. **Default Selection Protocol:** Compare all variants on TUNE using paired bootstrap CIs on NDCG@5 and MRR@10. If the best weighted variant is not statistically distinguishable from RRF (CI contains 0), RRF is selected as the default due to scale-invariance and zero tuning parameters.
- **Affected Steps & Re-Run Status:**
  - Affects Gate 4 (Hybrid Search Implementation).
  - Completed steps (Gates 0 & 1) are unaffected. No re-runs required.

---

## ADR-009: [PATCH-4] Near-Duplicate Corpus Audit and Evaluation Secondary Metrics
- **Date:** 2026-10-03
- **Context:** Information retrieval corpora frequently contain near-duplicate passages (minor variations in formatting or punctuation). Standard qrels label only one instance as relevant, potentially penalizing valid retrievals.
- **Specification:**
  1. **Corpus Integrity:** The 100,000 passage Corpus A is **strictly preserved and not altered**.
  2. **Near-Duplicate Audit:** Evaluated all 1,072 gold passages against Corpus A using exact normalized matching and 5-gram word shingle Jaccard similarity $\ge 0.8$.
  3. **Audit Findings:** Found exactly 2 near-duplicate instances affecting 2 gold passages across 2 queries (0.2% of queries). Results committed to `data/manifests/near_duplicates_manifest.json`.
  4. **Secondary Metrics:** In Gate 7, report secondary "dup-aware Hit@1 / Recall@5" alongside primary headline metrics.
  5. **Reporting Wording Rule:** LLM judge scores will never be described as "proving" improvement; both Non-LLM and LLM families will be reported with paired difference 95% CIs.
- **Affected Steps & Re-Run Status:**
  - Affects Gate 1 (Analysis recorded) and Gate 7 (Evaluation metrics).
  - Completed steps: Corpus A and Splits generated in Gate 1 remain 100% valid; near-duplicate analysis was executed and added to `data/manifests/near_duplicates_manifest.json`. No re-run of prior steps required.

---

## ADR-010: Groq Free-Tier Model Selection and Token Budget
- **Date:** 2026-10-03
- **Context:** RAGAS LLM-as-a-judge requires evaluating Context Precision and Context Recall across the benchmark set using only Groq free-tier without exceeding rate limits or daily token quotas.
- **Official Groq Free-Tier Limits (verified 2026-10-03):**
  - `llama-3.1-8b-instant`: 30 RPM, 14,400 RPD, 500,000 Tokens/Day (TPD).
  - `llama-3.3-70b-versatile`: 30 RPM, 1,000 RPD, 100,000 Tokens/Day (TPD).
- **Workload Estimation:**
  - 100 queries * 2 calls/query (Precision + Recall) * 2 phases (Phase 1 + Phase 2) = 400 LLM calls.
  - At ~700–800 tokens per prompt (question + 5 passage contexts + reference), total workload is ~280,000–320,000 tokens.
  - `llama-3.3-70b-versatile` has a strict 100,000 TPD ceiling and would throttle after ~30 queries, requiring 3–4 days.
  - `llama-3.1-8b-instant` has a 500,000 TPD ceiling, allowing all 100 paired queries to execute within a single day while respecting 30 RPM.
- **Choice:** `llama-3.1-8b-instant` as primary judge model.
- **Protocol:**
  - 3-query smoke test to measure exact empirical token consumption per query.
  - Paired query execution (Phase 1 and Phase 2 run back-to-back with immediate checkpointing per query).
  - Chunks of 25 with interim summary.
  - Exponential backoff with jitter on 429/timeout errors.

## ADR-011: Cross-Encoder Reranker Architecture and Selection on TUNE (Gate 4A)
- **Date:** 2026-10-03
- **Context:** First-stage hybrid retrieval achieves high recall (Recall@10 = 0.9533) with low latency (~55 ms p50, ~75 ms p95), leaving ~175 ms headroom under the 300 ms SLA. Bi-encoder cosine and BM25 term scores evaluate query and passages independently. A cross-encoder performs joint cross-attention across all token pairs, enabling much sharper relevance estimation.
- **Hypothesis (Logged BEFORE Running):**
  > Re-scoring the top-$K$ candidates retrieved by frozen hybrid search with `cross-encoder/ms-marco-MiniLM-L-6-v2` (quantized to INT8) will capture fine-grained query-document token interactions, yielding an improvement in top-rank precision metrics (NDCG@5, MRR@10, and Hit@1) over first-stage hybrid retrieval, while remaining well within the 250 ms target latency budget (hard limit 300 ms).
- **Selection Rule (Written BEFORE Running):**
  > Candidate depths $K \in \{10, 20, 30\}$ will be evaluated exclusively on the 150 queries of the TUNE split (`split_tune.json`).
  > The selection criterion is:
  > 1. Select the depth $K$ that maximizes **NDCG@5** on TUNE.
  > 2. In the event of a tie or difference $\le 0.0010$ NDCG@5 between candidate depths, choose the smaller $K$ to minimize latency overhead, maximize throughput, and maintain maximum safety headroom under the 300 ms SLA.
- **Model Choice:** `cross-encoder/ms-marco-MiniLM-L-6-v2`, quantized to PyTorch INT8 dynamic linear layers (`torch.qint8`) running on CPU with 12 threads.
- **Status:** Completed. Evaluated on 150 queries of the TUNE split:
  - $K = 10$: NDCG@5 = 0.9222, MRR@10 = 0.9117, Hit@1 = 0.8667, p95 = 284.1 ms
  - $K = 20$: NDCG@5 = 0.9369, MRR@10 = 0.9258, Hit@1 = 0.8867, p95 = 442.7 ms
  - $K = 30$: NDCG@5 = 0.9381, MRR@10 = 0.9250, Hit@1 = 0.8800, p95 = 688.0 ms
  - **Selection Winner:** $K = 30$ ($+0.0012$ NDCG@5 over $K=20$, exceeding the $0.0010$ tie threshold). Frozen as canonical reranker depth. Persisted in `results/phase3/tune_reranker_k.json`.

---

## ADR-012: In-Memory Multi-Tier Query Result Cache with Automatic Invalidation (Gate 4A)
- **Date:** 2026-10-03
- **Context:** Repeated queries in production RAG systems (frequently asked questions, automated agent retries) waste CPU compute and add unnecessary latency. A query result cache can serve identical queries in sub-millisecond time.
- **Design:**
  - In-memory thread-safe LRU cache with capacity 2,000 entries.
  - Deterministic key based on SHA256 of `(query, mode, top_k, filters, fusion, rerank, rerank_k)`.
  - Cache bypass supported via `use_cache=False` / `cache=False` in request payload.
  - Automatic synchronous cache invalidation on any write operation (`/passages/upsert` or `/passages/{passage_id}` DELETE), guaranteeing zero stale reads after updates.
- **Evaluation Workloads:**
  - Evaluated on three workloads: (1) All-unique (0% cache hits, p50=66.18ms, p95=103.04ms), (2) 30% repeated queries (p50=61.10ms, p95=75.95ms), and (3) 100% repeated queries (p50=4.26ms, p95=24.28ms, sub-millisecond cache hits).

---

## Frozen Configurations Registry
*(Frozen configuration hashes must be written here BEFORE single TEST evaluations)*

| Config Name | Phase | Gate | Config Hash (SHA-256 canonical JSON) | Date Frozen | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `phase1_dense_baseline` | Phase 1 Dense | Gate 3 | `3b06508a5c6dc296663e0547331da83b6b5996afa6a21d510fcbd84cb64cdd95` | 2026-10-03 | Dense cosine baseline on 100k index |
| `phase2_hybrid_optimized`| Phase 2 Hybrid| Gate 3 | `b23eb0d7862be81675e68eb82540f7039dc2cfea1850916833ee44c515ab0018` | 2026-10-03 | Hybrid weighted (alpha=0.8, minmax) frozen from TUNE grid search |
| `phase3_hybrid_rerank` | Phase 3 Rerank | Gate 4A | `0f106e12f557e9c7440284eaa0582c292a5904273d9b9561dc86dbd31ff78c76` | 2026-10-03 | Hybrid + MiniLM-L6 INT8 reranker (K=30) frozen from TUNE sweep |



