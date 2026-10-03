# Known Issues, Warnings, and Deviations

This document tracks system caveats, platform-specific warnings, contradictions, and recorded deviations with documented reasons and mitigations.

---

## 1. OneDrive Synchronization Warning
- **Issue:** The repository is located at `c:\Users\manis\OneDrive\Desktop\main_adrosonic`, which is under Microsoft OneDrive synchronization.
- **Risk:** Background file synchronizers (such as OneDrive or Dropbox) can hold transient file locks on newly created database files, SQLite journals, and Qdrant storage memory maps.
- **Mitigation:**
  - SQLite is configured with WAL (`journal_mode=WAL`) and a busy timeout (`busy_timeout=5000`).
  - Qdrant storage path is isolated in `data/qdrant_storage/`.
  - For maximum performance, users are advised to pause OneDrive synchronization during extensive ingest benchmarking.

---

## 2. Docker Unavailability on Host System
- **Issue:** Docker CLI is not installed or available in PATH on the Windows host.
- **Resolution:** As mandated by Gate 0 ("If Docker is unavailable, check whether an official native Qdrant binary exists for this OS by reading the official release page; use it if found"), the official native Qdrant server binary (`v1.19.1`, `qdrant-x86_64-pc-windows-msvc.zip`) was downloaded directly to `bin/qdrant.exe`.
- **Note:** Embedded mode was NOT used; this is a genuine native server running over HTTP (port 6333) and gRPC (port 6334).

---

## 3. Ragas 0.4.3 VertexAI Langchain Dependency Conflict
- **Issue:** The installed version of `ragas` (0.4.3) has an eager import in `ragas.llms.base` referencing `from langchain_community.chat_models.vertexai import ChatVertexAI`, which raises `ModuleNotFoundError` when VertexAI subpackage is not present.
- **Resolution:**
  - For Family A (Non-LLM Context Precision and Recall), implement exact standalone computation matching RAGAS formulas.
  - For Family B (LLM-based evaluation with Groq judge), invoke Groq API directly with RAGAS prompt templates, caching results to disk to ensure 100% reliability, deterministic retries, and free-tier compliance.

---

## 4. MS MARCO Passage Corpus Lacks Native Categories
- **Issue:** The problem statement references queries with constraints such as "documents from category X only", but MS MARCO passages have no inherent category labels.
- **Resolution:** As prescribed in Gate 2, categories are derived using MiniBatchKMeans ($k=15$, seeded) over passage embeddings, followed by c-TF-IDF cluster labeling. This is clearly disclosed as derived metadata in `docs/DATA.md` and the final report.
