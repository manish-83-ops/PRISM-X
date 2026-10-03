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

## Frozen Configurations Registry
*(Frozen configuration hashes must be written here BEFORE single TEST evaluations)*

| Config Name | Phase | Gate | Config Hash (SHA-256 canonical JSON) | Date Frozen | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *Pending Gate 3* | Phase 1 Dense | Gate 3 | *Pending* | - | Single TEST evaluation |
| *Pending Gate 4* | Phase 2 Hybrid| Gate 4 | *Pending* | - | Single TEST evaluation |
