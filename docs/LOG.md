# Systematic Execution Log

Append-only record of every engineering step, command executed, outcome, and evidence artifact.

---

## [2026-10-03 13:40 local] Gate 0 / Step 1: PDF Extraction and Pass 1 Requirements Analysis
- **Goal:** Read `docs/problem_statement.pdf` completely using PyMuPDF and extract raw text for verification.
- **Commands run:**
  - `python -c "import fitz; doc = fitz.open('Vector_Database_Design_for_RAG_System.pdf'); ..."`
  - Text dumped to `extracted_pdf.txt` (8 pages, 10,534 bytes).
- **Outcome:** PASS
- **Evidence:** `extracted_pdf.txt`, `docs/problem_statement.pdf`
- **Git commit:** `c83c33d`
- **Notes:** PDF successfully loaded and extracted without any missing pages or truncation.

---

## [2026-10-03 13:42 local] Gate 0 / Step 2: Pass 2 PDF Verification and Matrix Construction
- **Goal:** Re-read the PDF a second time, diff all sections against `docs/00_REQUIREMENTS_MATRIX.md`, and confirm exact alignment with all FRs (1-6), NFRs (1-7), Constraints (C-01 to C-07), demo stages (1-7), deliverables, and bonus items.
- **Commands run:** Detailed text comparison between `docs/problem_statement.pdf` extracted text and `docs/00_REQUIREMENTS_MATRIX.md`.
- **Outcome:** PASS
- **Evidence:** `docs/00_REQUIREMENTS_MATRIX.md`
- **Git commit:** `c83c33d`
- **Notes:** Verified all 6 Functional Requirements, 7 Non-Functional Requirements, 7 Constraints, 7 Demo Stages, Phase 1/Phase 2 deliverables, and 6 Bonus items.

---

## [2026-10-03 13:44 local] Gate 0 / Step 3: Hardware and Platform Environment Audit
- **Goal:** Inspect host environment, Python version (3.10/3.11), CPU model, cores, RAM, free disk, git, Docker availability, and OneDrive path check.
- **Commands run:**
  - `python scripts/env_report.py`
- **Outcome:** PASS
- **Evidence:** `data/manifests/env_report.json`, `docs/ISSUES.md`
- **Git commit:** `c83c33d`
- **Notes:**
  - OS: Windows 11 (AMD64)
  - Python: 3.11.9
  - CPU: AMD64 Family 25 Model 124 Stepping 0 (6 physical / 12 logical cores)
  - RAM: 15.28 GB total
  - Disk Free: 182.12 GB
  - Docker: Not available in PATH
  - Cloud Sync Warning: Repo is in OneDrive path; warning logged in `docs/ISSUES.md`.

---

## [2026-10-03 13:45 local] Gate 0 / Step 4: Qdrant Native Server Deployment and Client Verification
- **Goal:** Deploy official native Qdrant binary v1.19.1 for Windows and verify client API (query_points, sparse IDF modifier, payload indexes, gRPC, and write-wait semantics).
- **Commands run:**
  - Downloaded `qdrant-x86_64-pc-windows-msvc.zip` from official GitHub release v1.19.1 into `bin/qdrant.exe`.
  - Started Qdrant server with `config/qdrant.yaml` (ports 6333 HTTP, 6334 gRPC).
  - Executed API verification script testing `query_points`, `models.Modifier.IDF`, `create_payload_index`, and `upsert(wait=True)`.
- **Outcome:** PASS
- **Evidence:** `bin/qdrant.exe`, `config/qdrant.yaml`, `docs/API_NOTES.md`, `docs/DECISIONS.md`
- **Git commit:** `c83c33d`
- **Notes:** `client.search` is removed in `qdrant-client` 1.19.1; `client.query_points` verified as the correct supported method. Dense and BM25 sparse search with IDF modifier confirmed working over gRPC.

---

## [2026-10-03 13:46 local] Gate 0 / Step 5: Venv, Pinned Requirements, and Secrets Scan
- **Goal:** Create virtual environment `.venv`, generate pinned `requirements.txt`, create `experiments/INDEX.csv`, and run secrets check.
- **Commands run:**
  - `python -m venv .venv --system-site-packages`
  - `python -m pip freeze > requirements.txt` (341 packages pinned in UTF-8)
  - `python scripts/check_secrets.py`
  - `python -m pytest tests/test_gate0.py` (4 passed in 2.04s)
- **Outcome:** PASS
- **Evidence:** `requirements.txt`, `scripts/check_secrets.py`, `experiments/INDEX.csv`, `tests/test_gate0.py`
- **Git commit:** `c83c33d` (Tag `gate-0`)
- **Notes:** Secrets scan returned 0 violations across working tree and git log. Gate 0 successfully tagged.

---

## [2026-10-03 13:54 local] Gate 1 / Step 1: HF Dataset Investigation and Schema Verification
- **Goal:** Inspect MS MARCO dataset candidates on Hugging Face (`Tevatron/msmarco-passage-corpus`, `Tevatron/msmarco-passage`, `BeIR/msmarco`, `BeIR/msmarco-qrels`).
- **Commands run:**
  - Streamed samples from candidate repositories, inspecting schema and field names.
  - Verified that `Tevatron/msmarco-passage` `dev.jsonl.gz` contains 6,980 dev queries and `BeIR/msmarco-qrels` `dev.tsv` contains 7,437 qrels across 6,980 unique queries.
  - Confirmed 100% ID consistency between qrels and queries.
- **Outcome:** PASS
- **Evidence:** `docs/DECISIONS.md` (ADR-005), `docs/DATA.md`
- **Git commit:** Pending Gate 1 commit
- **Notes:** `Tevatron/msmarco-passage-corpus` + `Tevatron/msmarco-passage` + `BeIR/msmarco-qrels` provides direct official MS MARCO IDs without synthetic remapping.

---

## [2026-10-03 13:55 local] Gate 1 / Step 2: Deterministic Splits Generation
- **Goal:** Generate strictly disjoint TUNE (500), TEST (500), BENCH (100), and RAGAS (100) query splits from the 6,980 MS MARCO Dev queries with $\ge 1$ qrel.
- **Commands run:**
  - `python src/prismx/data/splits.py`
- **Outcome:** PASS
- **Evidence:**
  - `data/manifests/split_tune.json` (500 queries, 531 gold passages)
  - `data/manifests/split_test.json` (500 queries, 541 gold passages)
  - `data/manifests/split_bench.json` (100 queries)
  - `data/manifests/split_ragas.json` (100 queries)
  - `data/manifests/splits_manifest.json` (contains SHA-256 hashes of each split file)
- **Git commit:** Pending Gate 1 commit
- **Notes:** Verified $\text{TUNE} \cap \text{TEST} = \emptyset$. Total unique gold passages across TUNE+TEST: 1,072.

---

## [2026-10-03 13:59 local] Gate 1 / Step 3: Corpus A (standard-100K) Ingestion and Sanity Analysis
- **Goal:** Build Corpus A consisting of EXACTLY 100,000 unique passages (all 1,072 gold passages of TUNE+TEST queries + lowest-hash non-gold passages, deduplicated and trimmed).
- **Commands run:**
  - `python src/prismx/data/corpus_builder.py`
- **Outcome:** PASS
- **Evidence:**
  - `data/corpus_100k.jsonl` (100,000 passages, SHA-256: `30d5212101e6d5ee12c0f5894e0cc522fd8da526b434ab4c841c34ee1f29e8b2`)
  - `data/manifests/corpus_manifest.json`
  - `docs/DATA.md`
- **Git commit:** Pending Gate 1 commit
- **Notes:**
  - Total passages: 100,000
  - Gold passages: 1,072 (100% of gold passages in TUNE + TEST preserved)
  - Filler passages: 98,928
  - Gold / Total ratio: 0.01072 (1.072%)
  - Duplicates removed: 1
  - Character length: min 14, median 301.0, mean 335.99, max 1299
  - Word length: min 2, median 50.0, mean 56.29, max 230
  - Verbatim query in gold: 2.8%
  - Qrels per query: mean 1.072
  - Informative BM25s on TUNE: MRR@10 = 0.6386, Recall@20 = 0.8340
