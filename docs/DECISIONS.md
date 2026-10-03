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
- **Evidence:** Unit tests in `tests/test_determinism.py`.

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

## Frozen Configurations Registry
*(Frozen configuration hashes must be written here BEFORE single TEST evaluations)*

| Config Name | Phase | Gate | Config Hash (SHA-256 canonical JSON) | Date Frozen | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *Pending Gate 3* | Phase 1 Dense | Gate 3 | *Pending* | - | Single TEST evaluation |
| *Pending Gate 4* | Phase 2 Hybrid| Gate 4 | *Pending* | - | Single TEST evaluation |
