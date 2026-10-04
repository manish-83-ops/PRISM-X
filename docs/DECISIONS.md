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
  - Evaluated on three workloads: (1) All-unique in PRISM-X mode (0% cache hits, p50=66.18ms, p95=103.04ms on curated; p50=176.66ms, p95=250.50ms on raw), (2) 30% repeated queries: marked *invalid, cache not reset, re-run pending* (the initial Gate 5.5 script omitted cache invalidation before Workload 2, causing artificial 100% pre-warmed hits; script has been corrected, re-run pending idle machine authorization), and (3) 100% repeated queries (sub-millisecond cache hits, p50=3.93ms).

## ADR-013: Latency-Constrained Reranker Selection and Deadline Governor (Gate 4B)
- **Date:** 2026-10-03
- **Context:** While $K=30$ was selected under ADR-011 for unconstrained retrieval quality (NDCG@5 = 0.9381 on TUNE), its HTTP client p95 latency on this CPU reached 743.53 ms, significantly exceeding the 300 ms SLA.
- **Latency-Constrained Selection Rule (Written BEFORE Running):**
  > Among configurations whose p95 total latency on TUNE (HTTP path, idle machine) is $\le 250$ ms (with hard limit 280 ms), select the configuration that maximizes **NDCG@5**. Ties within $0.0010$ NDCG@5 go to the lower-latency configuration.
  > If NO reranker configuration achieves p95 $\le 250$ ms on TUNE, the default production system remains **Hybrid without reranking** (p95 = 104.24 ms), and reranking is designated as an optional, opt-in mode labeled "exceeds SLA".
- **Empirical Results on 150 TUNE Queries:**
  - $K = 5$: NDCG@5 = 0.9139, MRR@10 = 0.9089, Hit@1 = 0.8800, p50 = 102.5 ms, p95 = 163.0 ms, Truncation = 0.0% (Valid)
  - $K = 8$: NDCG@5 = 0.9229, MRR@10 = 0.9144, Hit@1 = 0.8800, p50 = 141.0 ms, p95 = 203.0 ms, Truncation = 0.7% (Valid)
  - $K = 10$: NDCG@5 = **0.9296**, MRR@10 = **0.9211**, Hit@1 = **0.8867**, p50 = 134.4 ms, p95 = **170.3 ms**, Truncation = 0.7% (**WINNER**)
  - $K = 15$: NDCG@5 = 0.9311, MRR@10 = 0.9208, Hit@1 = 0.8800, p50 = 205.6 ms, p95 = 269.4 ms, Truncation = 10.7% (FAILED SLA: > 250 ms)
  - $K = 20$: NDCG@5 = 0.9336, MRR@10 = 0.9208, Hit@1 = 0.8800, p50 = 229.5 ms, p95 = 266.4 ms, Truncation = 46.0% (FAILED SLA: > 250 ms)
- **Deadline Governor Evaluation on K=10:**
  - Deadline 150.0 ms: NDCG@5 = 0.9296, p95 = 198.4 ms, Truncation = 6.0%
  - Deadline 200.0 ms: NDCG@5 = **0.9305**, p95 = 219.2 ms, Truncation = 2.7% (**Selected Budget**)
  - Deadline 250.0 ms: NDCG@5 = 0.9296, p95 = 290.3 ms, Truncation = 0.7%
- **Selected Frozen Configuration:** Candidate depth $K = 10$, `max_length = 128`, PyTorch dynamic INT8 quantization, 8 threads, and Deadline Governor budget = $200.0$ ms.

---

## ADR-014: MS MARCO Human Reference Answer Ground Truth for RAGAS Evaluation
- **Date:** 2026-10-03
- **Context:** Evaluating RAGAS with long gold-passage text (~350 chars) creates reference-context confounding. The primary benchmark must evaluate against human-generated reference answers.
- **Answer-Selection Rule (Written BEFORE Running):**
  1. Use `wellFormedAnswers[0]` if present and valid ($>1$ word, informative).
  2. Otherwise use `answers[0]` if present and valid.
  3. Queries without valid answers ("No Answer Present", empty, or single-token "Yes"/"No") are excluded from the primary aggregate.
- **Audit Findings:** 96% (96/100) of BENCH queries have valid human answers; 4 queries were excluded for single-token responses.
- **Supersession Notice:** Prior Gate 3 $N=25$ run using passage text is marked "superseded".

---

## ADR-015: Hard-Distractor Stress Test Collection (`c100k_hard`) Construction Rule
- **Date:** 2026-10-03
- **Status:** Accepted (Written BEFORE building the stress-test collection)
- **Context:** The standard 100K index (`prismx_corpus`) is a closed-world benchmark containing guaranteed gold passages for evaluation queries with sparse qrels (~1.07 labeled passages per query). To stress-test retrieval precision and ranking robustness against unjudged semantic competitors, a separate collection `c100k_hard` is constructed. The original `prismx_corpus` collection remains completely untouched.
- **Construction Rule:**
  1. **Candidate Pool Extraction:** From the MS MARCO source pool (`data/raw/corpus.jsonl.gz`), sample non-100K passages matching semantic terms across the 150 seeded TUNE queries and 100 BENCH queries (250 total evaluation queries).
  2. **Strict Exclusions:** Exclude any passage present in `data/corpus_100k.jsonl` (zero overlap with standard corpus) and exclude any passage listed as a positive in `data/raw/dev_qrels.tsv` for any dev query (zero gold leakage).
  3. **Dense Neighbor Selection:** Encode candidate passages with `BAAI/bge-small-en-v1.5` and compute cosine similarity against each query's dense representation. For each query, select the top-20 dense-nearest passages as hard distractors.
  4. **Dedicated Collection Isolation:** Clone the 100,000 points of `prismx_corpus` into a new isolated Qdrant collection named `c100k_hard`. Upsert the deduplicated hard distractors (with BGE dense vectors and BM25 sparse vectors) into `c100k_hard`. Hydrate text into `data/text_store_hard.db`.
  5. **Single-Run Evaluation Policy:** Evaluate Dense, Hybrid, and Hybrid+Rerank ($K=10$) on BENCH queries exactly once, computing 10,000 paired bootstrap resamples. Run RAGAS answer-based evaluation (CP/CR) for Dense vs Hybrid on the frozen 25 queries using Groq `allam-2-7b`.
  6. **Reporting Discipline:** Report all stress-test metrics under the explicit label: *"Stress test (hard distractors, unlabeled neighbors may be valid answers; ID metrics are pessimistic)"*. Never merge stress-test metrics into headline benchmark tables. Do not use BENCH queries to tune or select parameters.

---

## Frozen Configurations Registry
*(Frozen configuration hashes must be written here BEFORE single TEST evaluations)*

---

## ADR-016: Raw Query-Centric Corpus (`c100k_raw`) Construction and Pre-Registration Rules
- **Date:** 2026-10-03
- **Status:** Accepted (Pre-registered and written BEFORE construction or evaluation)
- **Context:** The original corpus was constructed from the Tevatron MS MARCO corpus partition. To evaluate PRISMX on an uncurated, authentic multi-candidate distribution directly reflecting the query-centric nature of real-world search, a new collection `c100k_raw` is constructed directly from the raw MS MARCO v2.1 validation split (`microsoft/ms_marco`, configuration `v2.1`, validation split: 101,093 queries).
- **Construction Rule (B2):**
  1. **Source Dataset:** `microsoft/ms_marco` configuration `v2.1`, split `validation` (size: 101,093 queries).
  2. **Deterministic Sampling:** Queries are sampled sequentially with a fixed seed (`seed = 42`).
  3. **Candidate Ingestion:** For each sampled query, ingest all of its candidate passages (`sample["passages"]`).
  4. **Strict Deduplication:** Passages are deduplicated by normalized text (case-folded, whitespace-normalized).
  5. **Stopping Condition:** Add queries until the total unique passage count reaches $\ge 100,000$. Zero hand-inserted passages, zero synthetic fill, zero random text.
  6. **Passage Metadata:**
     - `passage_id`: Deterministic integer string ID.
     - `source`: Candidate URL (`url` from passage record).
     - `category`: `query_type` of the originating query (used for metadata filtering).
     - `originating_query_ids`: List of query IDs that brought this passage into the corpus.
- **Split Extraction & Gold Ground Truth (B3):**
  1. From the set of queries whose passages form `c100k_raw`, draw:
     - **TUNE Split:** 500 queries (fixed seed: `seed = 42`, disjoint from BENCH).
     - **BENCH Split:** 100 queries (fixed seed: `seed = 1337`, disjoint from TUNE).
  2. **Gold Labels:** Ground truth passages are defined strictly as candidates where `is_selected == 1`. The exact count of gold passages per query is recorded in evaluation manifests.
  3. **Reference Answers (ADR-014):** Evaluated against human reference answers (`wellFormedAnswers[0]` if valid, else `answers[0]`). Queries without valid human reference answers are excluded from primary RAGAS aggregates.
  4. **Caveat on Unselected Sibling Passages:** In MS MARCO, multiple candidate passages for a query may contain the correct answer even if annotators selected only one (`is_selected == 1`). Therefore, ID metrics (Hit@1, MRR@10, NDCG) on `c100k_raw` are inherently conservative/pessimistic estimates of true semantic relevance.
- **Pre-Registered Reporting Protocol (B4):**
  1. **Headline Corpus Policy:** The HEADLINE corpus for all final reporting and presentation is `c100k_raw`, regardless of whether scores are higher or lower than the curated corpus.
  2. **Curated Corpus Reporting:** The original `prismx_corpus` is reported strictly as a secondary comparison labeled *"Curated Corpus Baseline"*.
- **Frozen Models & Parameters Carryover (Deliberate Choice):**
  - Dense Model: `BAAI/bge-small-en-v1.5` (dim=384, normalize=True).
  - Sparse Lexical: Qdrant sparse vectors with dynamic BM25 IDF modifier ($k_1=1.2, b=0.75$).
  - Vector Index: Qdrant HNSW ($M=16, \text{ef\_construct}=100, \text{search\_ef}=64$).
  - Fusion: Weighted linear combination ($\alpha = 0.80$ dense, $0.20$ BM25 sparse, min-max normalized).
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` with dynamic INT8 quantization, `max_length = 128`, 8 torch threads, candidate depth $K = 10$.
  - **Deliberate Non-Retuning Choice:** As a deliberate methodology choice, parameters ($\alpha=0.80$, $K=10$, MiniLM-L6 INT8, `max_length=128`, rerank budget 200 ms) are carried over directly from the curated TUNE evaluation. We DO NOT re-tune $\alpha$ on `c100k_raw` and DO NOT re-select $K$. An $\alpha$ sensitivity table on $\{0.6, 0.7, 0.8, 0.9, 1.0\}$ is reported for informational purposes only and explicitly labeled "not used for selection".
- **Governor Update & Config Hash Invalidation (Gate 5.2):**
  - Introduced `total_deadline_ms = 250.0` (end-to-end request latency ceiling).
  - Cross-encoder stage receives `min(rerank_budget_ms=200.0, total_deadline_ms - elapsed_pre_rerank - 10.0ms safety)`.
  - This parameter addition updates the configuration schema and configuration hash.
  - **Latency Numbers Status:** All prior latency benchmark figures (e.g. 181.59 ms hybrid, 242.25 ms rerank K=10) are now categorized as *"curated corpus, previous governor"*. All headline latency numbers will be re-measured on `c100k_raw` when an idle-machine benchmark is authorized.
- **Evaluation Discipline:**
---

## ADR-017: Pre-Registered ANN Fidelity and Search-ef Selection Rule (Gate 5.3)
- **Date:** 2026-10-03
- **Status:** Accepted (Pre-registered and written BEFORE running ANN fidelity experiment)
- **Context:** An audit in Gate 5.2 (item 1c) revealed that apparent metric variations between curated and expanded collections were confounded by HNSW graph traversal nondeterminism at default `hnsw_ef = 64`. To guarantee retrieval fidelity and eliminate graph traversal truncation artifacts, PRISMX evaluates dense retrieval fidelity across candidate `hnsw_ef` values on the 500 TUNE queries on `c100k_raw`.
- **Pre-Registered Protocol & Selection Rule (Written BEFORE running):**
  1. **Candidate Set:** `hnsw_ef` $\in \{64, 128, 256\}$.
  2. **Ground Truth Baseline:** Exact brute-force dense cosine search (`exact = True` in Qdrant) over all 100,008 vectors.
  3. **Metrics Evaluated on 500 TUNE Queries:**
     - (a) Mean overlap of dense top-50 vs exact top-50: $\frac{1}{N}\sum_{q}\frac{|\text{top50}_{\text{ANN}}(q) \cap \text{top50}_{\text{exact}}(q)|}{50}$.
     - (b) Mean overlap of dense top-10 vs exact top-10: $\frac{1}{N}\sum_{q}\frac{|\text{top10}_{\text{ANN}}(q) \cap \text{top10}_{\text{exact}}(q)|}{10}$.
     - (c) Gold passage presence fraction in top-10 and top-50.
     - (d) Count of TUNE queries that lose their gold passage at `ef = 64` but retain it at the candidate `ef`.
  4. **Pre-Registered Decision Rule:**
     - **Choose the SMALLEST `ef` whose top-50 overlap vs exact is $\ge 0.99$ (99.0%).**
     - **If none of the candidates reaches $0.99$, choose `ef = 256` and state this explicitly.**
  5. **Serving Parameter Alignment:** If the chosen `ef` differs from current default (64), update `CONFIG.yaml`, schemas, and docs, record the resulting config hash change, and run the serving TUNE evaluation (Dense, Hybrid $\alpha=0.8$, Hybrid+Rerank $K=10$) at the chosen `ef`.
  6. **Zero BENCH Peeking:** BENCH queries are strictly excluded from the fidelity evaluation.
  7. **BM25 Channel Exactness Verification:** Qdrant sparse vectors with dynamic BM25 IDF modifier execute exact postings list traversal with inverted index dot-product scoring; sparse retrieval has zero graph approximation error.
- **Empirical Execution & Results (500 TUNE queries):**
  - **Exact Brute-Force Baseline:** Gold in Top-10 = 0.9540 (95.40%), Gold in Top-50 = 0.9900 (99.00%).
  - **`ef = 64`:** Top-50 Overlap = 0.9869 (98.69%), Top-10 Overlap = 0.9950 (99.50%), Gold Top-10 = 0.9540, Gold Top-50 = 0.9900. Status: `< 0.99` threshold.
  - **`ef = 128`:** Top-50 Overlap = 0.9954 (99.54%), Top-10 Overlap = 0.9978 (99.78%), Gold Top-10 = 0.9540, Gold Top-50 = 0.9900. Status: **$\ge 0.99$ (SELECTED per pre-registered rule)**.
  - **`ef = 256`:** Top-50 Overlap = 0.9986 (99.86%), Top-10 Overlap = 0.9992 (99.92%), Gold Top-10 = 0.9540, Gold Top-50 = 0.9900.
  - **Gold Loss Analysis:** 0 queries lost gold at `ef=64` vs exact search; `ef=128` provides 99.54% rank list overlap fidelity with exact search.
  - **Decision:** `chosen_ef = 128` (Smallest candidate with mean top-50 overlap $\ge 0.99$).
  - **Configuration Update:** `search_ef` updated from 64 to 128 in `CONFIG.yaml`, `src/prismx/schemas.py`, and `src/prismx/retrieve/service.py`.
  - **Config Hash Update:**
    - Prior hash: `508e053a1db7d271e068fb7f2c8815e05c02eb42d6e8a3d8f35f91c561e9961a`
    - Updated hash: `8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf`

---

## ADR-018: Pre-Registered Serving Default Mode Rule, Phase Naming, and Scientific Reporting Protocol (Gate 5.4)
- **Date:** 2026-10-03
- **Status:** Accepted (Pre-registered and written BEFORE executing single-run BENCH evaluation)
- **Context:** Following index build and ANN fidelity calibration (`chosen_ef = 128`, config hash `8e1000d5...`), the final 100-query BENCH split (`data/c100k_raw/bench_raw_100.json`) must be evaluated exactly once. To eliminate any possibility of post-hoc rationalization, all decision criteria, phase definitions, and reporting rules are pre-registered here.

### 1. Default-Mode Decision Rule (Written BEFORE running BENCH):
- The system default mode (`retrieval.default_mode` in `CONFIG.yaml` and serving endpoints) shall be set to **`"prismx"`** (hybrid + rerank $K=10$, `total_deadline_ms = 250.0`) IF AND ONLY IF:
  1. The single-run BENCH paired-bootstrap 95% confidence interval of (Rerank minus Hybrid) on **MRR@10** strictly excludes $0$ ($\Delta\text{MRR@10} > 0$ with $p < 0.05$); **AND**
  2. The idle-machine 100-query HTTP $p95$ end-to-end latency of `prismx` is $\le 250.0\text{ ms}$ (to be measured on an authorized idle-machine benchmark).
- **Fallback Rule:** If either condition fails, the system default mode shall remain **`"hybrid"`** ($\alpha = 0.8$), and `prismx` will be reported and presented in the UI as the highlighted optional high-precision rerank mode.

### 2. Phase Naming for Reports and UI:
- **Phase 1:** Dense Retrieval (`BAAI/bge-small-en-v1.5`, `dim=384`, `search_ef=128`).
- **Phase 2:** The selected serving default mode established under Rule 1 above.
- **Hybrid-Only Retrieval:** Explicitly presented as an ablation / first-stage fusion baseline row when Phase 2 is `prismx`.

### 3. Scientific Wording and Methodological Caveats:
- **Improvement Criterion:** A performance delta $\Delta$ may only be characterized as an "improvement" if the 95% bootstrap confidence interval of the paired difference strictly excludes $0$. If the interval includes $0$, the delta must strictly be characterized as *"directional, not significant"*.
- **Fusion Weight Characterization:** Parameter $\alpha = 0.80$ carried over from curated evaluation must NEVER be described as "validated" on `c100k_raw`; it must be described as *"not worse than alternatives"* based on the informational sensitivity sweep.
- **Unselected Sibling Passages:** Unselected sibling passages from the same MS MARCO originating query must NEVER be asserted to "contain relevant content". They are formally characterized as *"unjudged passages that may or may not be valid answers"*, reflecting MS MARCO sparse labeling where annotators stopped after marking one passage.
- **Cross-Encoder Annotator Preference Caveat:** Because `cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on MS MARCO train labels, ID-metric gains (MRR@10, NDCG) on MS MARCO validation queries may partly reflect learned annotator-style selection preferences in addition to semantic relevance.

### 4. Groq Free-Tier API Rate Limits (Official Console Specification):
- **Model:** `llama-3.3-70b-versatile`
- **Requests Per Minute (RPM):** **30 RPM**
- **Requests Per Day (RPD):** **1,000 RPD**
- **Tokens Per Minute (TPM):** **12,000 TPM** (12k)
- **Tokens Per Day (TPD):** **100,000 tokens/day** (100k)
- **Rate Limit Safeguards:**
  - Automated runner must throttle requests to $\le 20\text{ RPM}$ to prevent bursts exceeding 12k TPM.
  - Daily automatic safety cutoff initially set at 90% of daily tokens (**90,000 tokens/day**).
  - **Gate 5.5 Daily Token Cap Adjustment:** Adjusted daily cutoff from 90,000 to **96,000 tokens/day** (reason: with empirical consumption of ~1,215 tokens per evaluation, the 90,000 cap yielded 24 complete paired queries, strictly below the pre-registered minimum primary target of $N=25$; a 96,000 cap ensures at least 25-26 complete paired queries fit on Day 1 while remaining safely below the 100,000 free-tier daily ceiling).
  - Explicit exponential backoff handling HTTP 429 response codes.

### 5. ADR-018 Evaluation Outcome (Gate 5.5 Idle-Machine Latency Benchmark):
- **Condition (i) [Quality]:** $\Delta\text{MRR@10} = +0.0583$, 95% CI $[+0.0015, +0.1165]$ (excludes $0$, statistically significant at $p < 0.05$) $\rightarrow$ **MET**.
- **Condition (ii) [Latency]:** Idle-machine 100-query HTTP $p95 = 306.39\text{ ms} > 250.0\text{ ms}$ SLA target ceiling $\rightarrow$ **NOT MET**.
- **Mechanical Decision:** Per Fallback Rule, system default mode remains **`"hybrid"`** ($\alpha = 0.80$). `prismx` is designated as the highlighted optional high-precision rerank mode.
- **Config Hash:** Unchanged at `8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf`.

---

## ADR-020: Post-Freeze Serving Governor Hard Boundary (PROPOSED, NOT APPLIED)
- **Date:** 2026-10-04
- **Status:** **PROPOSED, NOT APPLIED**
- **Context:** Diagnostic analysis of live UI queries (Gate 5.8) revealed that when the cross-encoder runs on a non-idle machine, a single micro-batch of size 5 can consume $>250\text{ ms}$ of CPU time. Because the deadline governor checks elapsed time only *between* batches, the request overshoots the 250 ms target ceiling before truncation can occur.
- **Proposed Technical Solution:**
  1. Reduce reranker micro-batch size from 5 to 2 (or 1), allowing fine-grained preemption every ~35-45 ms.
  2. Pass true total request start timestamp (`t_total_start`) directly into `rerank()`, bounding total end-to-end request duration rather than rerank stage elapsed time.
  3. Include query encoding, dense search, sparse search, and SQLite fetch time explicitly in the deadline check:
     $$\text{if } (t_{\text{now}} - t_{\text{total\_start}}) \ge (\text{total\_deadline\_ms} - 15.0\text{ ms}): \text{break}$$
- **Reason Not Applied:**
  - Config hash `8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf` is strictly frozen.
  - Applying this change post-freeze would alter the serving governor behavior that produced the benchmarked numbers on BENCH.
  - Under ADR-018, PRISM-X is designated as the **optional** high-precision mode, while Hybrid ($p95 = 89.02\text{ ms}$) is the SLA-compliant serving default.
  - If approved for future deployment, this change requires an offline parity audit (rank correlation $\rho \ge 0.999$, Top-5 agreement = 100%) and truncation-rate verification before activation.

---

## Proposal B Disposition: Runtime Speedups (FUTURE WORK, NOT APPLIED)
- **Status:** **RECORDED AS FUTURE WORK, NOT APPLIED**
- **Evaluated Candidates:**
  1. **ONNX Runtime INT8 Reranker:** Exporting `ms-marco-MiniLM-L-6-v2` with dynamic INT8 quantization and ORT graph optimizations (estimated 1.8x–2.4x speedup on x86 CPU).
  2. **Thread Affinity / Execution Pool Tuning:** Binding PyTorch intra-op threads to physical cores (6 threads) to prevent hyperthreading contention.
  3. **OS Power Scheme Optimization:** Enforcing high-performance CPU governor to prevent idle frequency drops.
- **Disposition Policy:** Any runtime acceleration that modifies the inference engine or threading model requires a verified ranking-parity check against the PyTorch baseline (Spearman rank correlation $\ge 0.999$, 100% Top-5 agreement on 500 TUNE queries) and formal ADR approval before use. Config remains frozen.

---

## ADR-019: Pre-Registered RAGAS Frozen-50 LLM Judge Protocol (Gate 5.12 / Gate 6)
- **Status:** **APPROVED & PRE-REGISTERED (Pre-Execution)**
- **Date:** 2026-10-04
- **Context:** Formal RAGAS evaluation on `data/manifests/frozen_ragas_bench_raw_50.json` (N=50 query split).
- **Judge Model Accessibility Hierarchy:**
  1. Primary: `llama-3.3-70b-versatile`
  2. Fallback: `openai/gpt-oss-120b` (low reasoning effort if supported)
  3. Hard Stop Rule: If neither is accessible via `GET https://api.groq.com/openai/v1/models`, stop immediately and report available models.
- **Inference & Scoring Parameters:**
  - Temperature = 0 where supported.
  - Contexts: Top-5 retrieved contexts per mode (`dense`, `hybrid`, `hybrid_rerank`).
  - Metrics: Context Precision (CP) and Context Recall (CR) only.
  - Reference: `wellFormedAnswers[0]` if present and non-empty, else `answers[0]`.
- **Token Budget & Quota Management:**
  - Protocol: Query-major evaluation across all 3 modes.
  - Calibration: The first 2 queries (part of real run, nothing discarded) measure tokens per query-triple.
  - Daily token cap is set so that at least $N \ge 25$ complete paired queries fit under the account's daily limit with a 4,000-token safety margin. If projected $N < 25$, execution halts.
- **Retry & Failure Handling:**
  - On HTTP 429 / 5xx: Exponential backoff with up to 3 retries of the identical call; never rephrase prompts or change the judge.
  - Parse failure or NaN counts as a failed query and is excluded from all three phases (paired exclusion) and listed with reason.
  - Checkpoint persisted after every query.
- **Reporting & Claim Standards:**
  - Use "improvement" only if the 95% bootstrap confidence interval strictly excludes 0; otherwise report "no measurable difference".
  - Prior N=25 exploratory runs marked superseded.

---

## ADR-021: Speed-Only Serving Optimization Protocol, Parity Gates, and Default Mode Decision (Gate 6)
- **Status:** **APPROVED & PRE-REGISTERED (Pre-Execution)**
- **Date:** 2026-10-04
- **Branch:** `serving-opt` created from `final-system` (tagged `v1-frozen` at commit `c903499`).
- **Scope & Constraints:**
  - **Allowed Changes (Speed-Only):** Thread/env hygiene, query-encoder runtime, Qdrant client protocol and call pattern, SQLite access, reranker runtime/batching/max_length, predictive governor v2, embedding cache.
  - **Strictly NOT Allowed:** Models, weights, candidate depth $K=10$, fusion weight $\alpha=0.80$, Top-50 candidate limits, $ef=128$, fusion method, candidate set, index configuration, relevance thresholds.
- **Pre-Registered Parity Gates (Evaluated against v1 on 500 TUNE queries):**
  1. Query embedding cosine similarity $\ge 0.9999$ for every query ($100\%$).
  2. Dense Top-50 candidate overlap $\ge 99.9\%$.
  3. Reranker final top-5 set identical on $\ge 99.0\%$ of queries.
  4. Top-1 identical on $\ge 99.0\%$ of queries.
  5. Spearman rank correlation of rerank scores $\ge 0.99$.
  - *Gate Enforcement Rule:* A component failing any parity gate is rejected (try next variant) and logged.
- **Governor v2 Serving Parameter:**
  - `total_deadline_ms = 230` ms (server-side predictive budget) to guarantee HTTP client $p95 \le 250$ ms.
  - Bounded from request start (including encode, retrieval, fetch).
  - Predictive condition: Run micro-batch only if $\text{elapsed} + 1.2 \times \text{batch\_EMA} \le \text{total\_deadline\_ms} - 10$.
  - `governor_state` reported as `normal`, `truncated`, or `skipped_budget`.
- **Decision Rule (Unchanged ADR-018):**
  - System default mode becomes `prismx` IF AND ONLY IF idle HTTP $p95 \le 250.0\text{ ms}$ in **BOTH** of two independent official runs (the higher p95 counts).
  - Otherwise, default remains `hybrid` ($\alpha=0.80$).
  - Exactly one optimization round; no third attempt.
- **Dual Hash Verification:**
  - `semantic_hash`: Must remain strictly identical to v1 (`8e1000d561cb1e7dc190722897a59cd52d28ba2284ef2fb766d6081c082eabdf`).
  - `serving_hash`: Computed over new serving parameters (`total_deadline_ms`, batching, runtime flags).

---

## ADR-022: Anytime Cascade and Request-Level Predictive Governor (Gate 12 Part 1)
- **Status:** **APPROVED & PRE-REGISTERED (Pre-Execution)**
- **Date:** 2026-10-04
- **Branch:** `gate12-cascade`
- **Context:** Reranking with a Cross-Encoder ($K=10$) on CPU causes tail latency violations ($p95 = 306.39$ ms in v1) when earlier retrieval or serialization stages consume variable time. A stage-local deadline fails because it ignores the actual request wall-clock elapsed time.
- **Decision:**
  1. **Request-Level Clock:** Measure elapsed time from request arrival at the ASGI middleware level ($t_0$). Compute dynamic remaining budget:
     $$\text{remaining} = \text{total\_deadline\_ms}(230) - (t - t_0) - \text{reserve\_ms}$$
     where $\text{reserve\_ms}$ is the measured p95 of text hydration and JSON serialization.
  2. **Anytime Stage Semantics:**
     - Stage 1 (Hybrid dual-vector retrieval with concurrent dense encode and sparse search) always completes and serves as guaranteed fallback.
     - Stage 3 Cross-Encoder reranks top-$K_{\text{eff}}$ candidates in micro-batches (2–3 pairs) with hard deadline checks between batches.
     - Dynamic candidate clamping: $K_{\text{eff}} = \text{clamp}(\lfloor \text{remaining} / \text{per\_pair\_ms} \rfloor, 0, K)$ using rolling median per-pair cost calibrated at startup (20 warm pairs). Candidates beyond $K_{\text{eff}}$ maintain first-stage hybrid ranking below the scored subset.
     - States: `normal`, `truncated`, `skipped_budget`.
  3. **Runtime Engine Optimization:**
     - Evaluate ONNX Runtime FP32 and dynamic INT8 quantization (using physical core pinning, `inter_op=1`, `allow_spinning=0`, length-sorted dynamic padding, max_length {128, 96}).
- **Pre-Registered Parity Gates (Evaluated against PyTorch FP32 on all 500 TUNE queries):**
  - Top-1 passage identity agreement $\ge 0.97$.
  - Top-5 candidate set agreement $\ge 0.98$.
  - Paired $\Delta \text{NDCG@5}$ and $\Delta \text{Hit@1}$ satisfy $|\Delta| \le 0.005$ with $95\%$ bootstrap confidence interval containing 0.
  - *Decision Rule:* Adopt the fastest ONNX variant passing all parity gates. If an optimized variant fails any gate, reject it and report exact metrics.

---

## ADR-023: Evaluation Power, Zero-Token Analysis, and Persistent Verdict Ledger (Gate 12 Part 2)
- **Status:** **APPROVED & PRE-REGISTERED (Pre-Execution)**
- **Date:** 2026-10-04
- **Branch:** `gate12-cascade`
- **Context:** Downstream RAGAS evaluation with LLM judges is constrained by API rate limits, daily token caps, and sample size variance ($N=50$ bootstrap CIs cross zero). Redundant LLM calls on identical passage-query pairs waste tokens.
- **Decision:**
  1. **Audit Installed Ragas Semantics:** Read `ragas` source code to verify whether Context Precision evaluates contexts independently or jointly, record the exact aggregation denominator, and inspect Context Recall inputs.
  2. **Zero-Token Analysis on Stored Data:** Recompute Top-1 identity, Top-5 Jaccard overlap, lexical-only vs dense-only contributions, CP headroom, and the top-10 rerank degradation queries using existing benchmark files without API calls.
  3. **Persistent Verdict Ledger:** Maintain `results/ragas/verdict_ledger.jsonl` keyed by $(qid, \text{passage\_id}, \text{judge}, \text{prompt\_hash}, \text{ragas\_version})$. When evaluating a mode, retrieve existing verdicts from the ledger; call the judge only for novel pairs.
  4. **Pre-Registered Claim Rule:** Primary RAGAS evaluation remains pre-registered at $N=50$. Any extended evaluation ($N > 50$) enabled by ledger token savings will be reported in a separate, dedicated section.

---

## ADR-024: Architecture Integrity: Version-Stamped Cache, Dual-Write Outbox, and Sparse Hash Audit (Gate 12 Part 3)
- **Status:** **APPROVED & PRE-REGISTERED (Pre-Execution)**
- **Date:** 2026-10-04
- **Branch:** `gate12-cascade`
- **Context:** Global cache invalidation on write thrashes cache hit rates. Dual writes across Qdrant and SQLite risk consistency drift if a crash occurs mid-update. Sparse vector hashing may suffer token collisions.
- **Decision:**
  1. **Version-Stamped Cache Keys:** Cache keys incorporate `corpus_version` stored in SQLite metadata. Writes increment `corpus_version`, rendering all previous cache entries invalid in $O(1)$ without memory wipes. Ongoing read requests that started before a write complete under the old version key. Deletions evict via a reverse index mapping `passage_id \to \text{cache\_keys}`.
  2. **Transactional Dual-Write Outbox:** SQLite serves as authoritative source of truth. Mutations write document data and an outbox record (`pending_ops`) in a single SQLite transaction, followed by idempotent Qdrant write, and outbox resolution. Startup sequence replays unapplied operations.
  3. **Sparse Token Hash Collision Audit:** Audit 32-bit Murmur/SHA token hash space over MS MARCO vocabulary to determine exact collision frequency and affected query tokens on TUNE.

---

## ADR-025: ColBERT Late-Interaction Multi-Vector Reranking Stage (Gate 12 Part 4)
- **Status:** **PARKED (Pre-Registered, Execution Blocked until "GO COLBERT")**
- **Date:** 2026-10-04
- **Branch:** `gate12-cascade`
- **Context:** Cross-encoders compute $O(K \times L^2)$ all-to-all attention. Late-interaction ColBERT models (`answerai-colbert-small-v1`) pre-compute document token embeddings and compute query-document similarity via MaxSim in $O(L_Q \times L_D)$, running in $<25$ ms.
- **Decision Rule (Pre-Registered):**
  - Verify license, availability, and model size via HuggingFace API.
  - Do NOT train or execute index build until explicit user confirmation (`GO COLBERT`).
  - Pre-registered adoption rule: Adopt ColBERT as stage-2 intermediate filter IF AND ONLY IF on TUNE it achieves NDCG@5 non-inferior to CE (95% CI lower bound $\ge -0.02$) at materially lower p95 latency. Any negative result will be published without cherry-picking.

---

## ADR-026: Gate 13 System Hardening — Governor Honesty, Telemetry Labels, Warm-up Optimization, and Parity Verification (Gate 13 Step 1)
- **Status:** **APPROVED & PRE-REGISTERED (Pre-Execution)**
- **Date:** 2026-10-04
- **Branch:** `gate12-cascade`
- **Context:**
  The Anytime Cascade architecture (ADR-022) introduces dynamic candidate clamping and micro-batching to bound tail latency. For production compliance under Gate 13, all telemetry, UI labels, and warm-up paths must adhere to rigorous honesty, transparent reporting, and pre-registered parity gates.
- **Decision:**
  1. **Governor Honesty & Request-Level Telemetry:**
     - The FastAPI ASGI middleware captures request arrival time $t_0$ at the earliest network boundary.
     - Telemetry breakdown logs all discrete stages (`encode`, `dense`, `sparse`, `fusion`, `fetch_text`, `rerank`, `total`).
     - Response schema reports: `governor_state` (`normal`, `truncated`, `skipped_budget`), `stage_reached` (`stage1_hybrid`, `stage3_rerank`), `candidates_scored`, `K_requested`, `per_pair_ms`, and `effective_mode`.
     - When request deadline budget is insufficient, the system gracefully degrades to first-stage hybrid results. Queries with `candidates_scored < K_requested` are tracked separately and displayed honestly without concealing degradation.
  2. **UI & Demonstration Label Accuracy:**
     - UI reflects governor decisions in real time with distinct badges:
       - Normal: `✓ Reranked K of K (normal, ~Xms/pair)`
       - Truncated: `⚠️ Reranked N of K (truncated by deadline governor, ~Xms/pair)`
       - Skipped: `⚠️ Degraded to hybrid (budget exhausted, scored 0 of K)`
     - SLA compliance claim is attached strictly to the default serving mode (**Hybrid**, $p95 = 89.02\text{ ms} < 300\text{ ms}$). The PRISM-X optional mode ($p95 = 306.39\text{ ms}$) is never claimed to meet SLA unless confirmed by official idle benchmark under ADR-021.
  3. **Fetch & Server Warm-up Hygiene:**
     - Server startup sequence executes an end-to-end pre-flight warm-up pass:
       - Dense encoder: encode 2 warm-up text strings.
       - Sparse tokenizer: hash & tokenize 2 warm-up queries.
       - Cross-encoder: 1 dummy micro-batch of candidate pairs through ONNX runtime session.
       - SQLite text store: touch mmap header and run sample passage hydration query.
     - This guarantees that steady-state benchmark queries do not suffer from cold JIT/library compilation overhead.
  4. **Latency Reporting Protocols:**
     - In accordance with Gate 13 Step 1 & 4j, latency benchmarks report two distinct rows:
       - **Row 1:** First 100 queries without discarding any (cold start included from fresh server start).
       - **Row 2:** 100 queries with 20 warm-ups discarded.
     - Numbers remain marked `PENDING` until the authorized idle session in Step 3.
  5. **Parity Gate Enforcement:**
     - Any runtime engine changes (e.g. ONNX Runtime FP32/INT8) must satisfy pre-registered parity gates against the baseline PyTorch FP32 cross-encoder on all 500 TUNE queries:
       - Top-1 passage agreement $\ge 0.97$.
       - Top-5 candidate set agreement $\ge 0.98$.
       - Paired $|\Delta \text{NDCG@5}| \le 0.005$ with $95\%$ bootstrap CI containing 0.
     - If a candidate fails any gate, it is rejected and documented. A strict 60-minute time limit applies to Step 2 optimization.

---

## ADR-027: Gate 15 Production-Readiness Serving Layer Upgrades, Observability, and Overhead Gates
- **Status:** **APPROVED & PRE-REGISTERED (Pre-Execution)**
- **Date:** 2026-10-04
- **Branch:** `gate15-production`
- **Context:**
  Gate 15 hardens the FastAPI serving architecture for production deployment without modifying retrieval models, weights, thresholds, indices, or benchmark stores:
  - Observability: Standardized `/health`, `/ready` (verifying encoder, reranker, and Qdrant readiness with corpus count and config hashes), `/metrics` (Prometheus text exposition for request counters, latency histograms per stage, governor states, and cache hit ratios), `X-Request-ID` tracing, `Server-Timing` headers, and non-blocking asynchronous JSON structured logging via queue handlers.
  - Security: Bearer-token authentication for write endpoints (`WRITE_TOKEN`), `PUBLIC_DEMO` flag to disable mutable actions, sliding-window rate limiting on `/search` and `/answer`, request body size limits, max query string length bounds, strict CORS allowlist, HTTP security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`), explicit client timeouts to Qdrant and LLM providers, and sanitized error responses (no raw Python stack traces leaked).
  - CI & Packaging: Automated GitHub Actions CI workflow (`.github/workflows/ci.yml`), multi-stage non-root `Dockerfile`, `docker-compose.yml` with native healthchecks, and an operational `Makefile`.
  - Open Source Licensing: Enterprise Apache-2.0 license, comprehensive `THIRD_PARTY.md` attribution, and MS MARCO non-commercial dataset terms documented in `docs/BUSINESS_CASE.md`.
- **Pre-Registered Hypotheses:**
  1. **Retrieval Parity:** Serving layer instrumentation, headers, and security middleware introduce zero divergence in ranking. For dense and hybrid search on all 100 BENCH queries, top-10 passage IDs and scores must match the golden baseline identically (scores within $10^{-6}$). For PRISM-X reranked mode, top-10 IDs must match identically for all queries where `candidates_scored == K`.
  2. **Serving Overhead Bounds:** On the default serving mode (Hybrid), the combined overhead of all serving upgrades (ASGI middleware, request ID generation, Server-Timing header, Prometheus metrics recording, and structured logging) shall add:
     - $\le 1.0\text{ ms}$ to $p50$ latency
     - $\le 2.0\text{ ms}$ to $p95$ latency
- **Mechanical Rule:**
  If the A6 overhead gate fails ($>1.0\text{ ms}$ $\Delta p50$ or $>2.0\text{ ms}$ $\Delta p95$ against baseline on the default mode), the offending serving component shall be immediately disabled by default in configuration, flagged with a warning, and documented in `docs/ISSUES.md` and `docs/PDF_COMPLIANCE.md`. No ad-hoc parameter tuning or benchmark manipulation is permitted.
- **Idle Gate Protocol:**
  Timing runs for the overhead gate are strictly deferred until explicit `"IDLE CONFIRMED"` authorization from the user.
