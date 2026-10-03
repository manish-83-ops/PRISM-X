# Requirements Traceability Matrix

Source of Truth: `docs/problem_statement.pdf` (`Vector_Database_Design_for_RAG_System.pdf`, 8 pages).
Verification Status: Read completely in Pass 1 and verified against text in Pass 2 on 2026-10-03.

| ID | Requirement Paraphrase | Status | Evidence Path | Date |
| :--- | :--- | :--- | :--- | :--- |
| **FR-1** | Download & parse MS MARCO from HF; index $\ge 100,000$ passages in vector DB; generate sentence-transformer embeddings; store metadata (passage_id, source, category); index time $<2$ hrs on CPU | NOT STARTED | `data/manifests/corpus_manifest.json`, `results/ingest_stats.json` | 2026-10-03 |
| **FR-2** | Cosine/dot-product search over passage index; return top-5 passages for query; log response times for evaluation queries; compute & log RAGAS precision & recall on $\ge 20$ queries | NOT STARTED | `results/phase1/metrics.json`, `results/phase1/latency_dense.csv` | 2026-10-03 |
| **FR-3** | Implement BM25 sparse search alongside dense vector retrieval; combine scores using RRF or weighted combination; demonstrably improve RAGAS scores over Phase 1; document fusion method and configurable weights | NOT STARTED | `docs/FUSION.md`, `results/phase2/metrics.json`, `results/phase2/ablation.json` | 2026-10-03 |
| **FR-4** | Support $\ge 1$ filter dimension on query (category, source); apply pre-retrieval at vector DB level, not post-retrieval; demonstrate filtered query in demo | NOT STARTED | `tests/test_filters.py`, `results/phase2/filter_demo.json` | 2026-10-03 |
| **FR-5** | Live updates: upsert (add/update without full reindex); delete (remove passage by ID); verify both in API/interface | NOT STARTED | `tests/test_live_updates.py`, `results/phase2/live_update_demo.json` | 2026-10-03 |
| **FR-6** | Query interface: accept natural language question, return top-5 with scores, display retrieval method (dense/hybrid) & per-passage scores, allow toggle | DELEGATED | UI delegated to teammates; API & CLI implemented in `src/prismx/api/`, `src/prismx/__main__.py` | 2026-10-03 |
| **NFR-1** | Context Precision $> 0.75$ in Phase 2 (Phase 1 baseline recorded for comparison) | NOT STARTED | `results/phase2/ragas_eval.json`, `reports/BENCHMARK_REPORT.md` | 2026-10-03 |
| **NFR-2** | Context Recall $> 0.70$ in Phase 2 | NOT STARTED | `results/phase2/ragas_eval.json`, `reports/BENCHMARK_REPORT.md` | 2026-10-03 |
| **NFR-3** | Query latency: p95 latency $< 300$ ms, measured across 100 consecutive queries (internal target $\le 250$ ms) | NOT STARTED | `results/phase2/benchmark_summary.json`, `results/phase2/latency_hybrid.csv` | 2026-10-03 |
| **NFR-4** | Index scale: minimum 100,000 passages indexed (500K is bonus) | NOT STARTED | `data/manifests/corpus_manifest.json` | 2026-10-03 |
| **NFR-5** | Free-tier only: no paid API subscriptions; Groq free tier & HuggingFace permitted | IN PROGRESS | Verified in `scripts/check_secrets.py`, `src/prismx/eval/ragas_eval.py` | 2026-10-03 |
| **NFR-6** | Reproducibility: GitHub repo with README and setup steps; evaluators can clone and run | NOT STARTED | `README.md`, verified by Gate 9 clean-clone test `results/clean_clone_test.txt` | 2026-10-03 |
| **NFR-7** | Benchmarking report: Markdown/PDF comparing Phase 1 vs Phase 2 RAGAS scores and latency logs | NOT STARTED | `reports/BENCHMARK_REPORT.md` | 2026-10-03 |
| **C-01** | Free-tier tools only (no paid GPU clusters, no enterprise API keys) | IN PROGRESS | Verified: zero paid APIs, Groq free tier only | 2026-10-03 |
| **C-02** | RAGAS evaluation mandatory for both Phase 1 and Phase 2 | NOT STARTED | `results/phase1/ragas_eval.json`, `results/phase2/ragas_eval.json` | 2026-10-03 |
| **C-03** | Hybrid search required for Phase 2 (dense-only disqualifies hybrid scoring) | NOT STARTED | `src/prismx/retrieve/hybrid.py`, `docs/FUSION.md` | 2026-10-03 |
| **C-04** | Minimum dataset size $\ge 100,000$ MS MARCO passages | NOT STARTED | `data/manifests/corpus_manifest.json` | 2026-10-03 |
| **C-05** | Latency benchmark required: 100 consecutive queries, p95 logged, no self-estimates | NOT STARTED | `results/phase1/latency_dense.csv`, `results/phase2/latency_hybrid.csv` | 2026-10-03 |
| **C-06** | GitHub repo required: clonable and runnable | NOT STARTED | Tested in Gate 9 clean-clone test; push performed by user | 2026-10-03 |
| **C-07** | Phase 1 baseline required: logged Phase 1 RAGAS baseline mandatory | NOT STARTED | `results/phase1/ragas_eval.json` | 2026-10-03 |
| **Demo-1** | Live query (dense): top-5 retrieved passages with scores from Phase 1 | DELEGATED | UI delegated; API endpoint `POST /search` (`mode: dense`) supports it | 2026-10-03 |
| **Demo-2** | RAGAS baseline: display Phase 1 Context Precision and Recall from logged run | DELEGATED | UI delegated; API endpoint `GET /eval/latest` supports it | 2026-10-03 |
| **Demo-3** | Hybrid search in action: run same query, show fused results vs Phase 1 | DELEGATED | UI delegated; API endpoint `POST /search` (`mode: hybrid`) supports it | 2026-10-03 |
| **Demo-4** | Phase 1 vs Phase 2 comparison: side-by-side RAGAS score improvements | DELEGATED | UI delegated; `reports/BENCHMARK_REPORT.md` and API provide data | 2026-10-03 |
| **Demo-5** | Latency benchmark: 100 consecutive queries log, show p95 $< 300$ ms | DELEGATED | UI delegated; API endpoint `GET /bench/latest` and report support it | 2026-10-03 |
| **Demo-6** | Metadata-filtered query: demonstrate query with category filter changing result set | DELEGATED | UI delegated; API `POST /search` with `filters` supports it | 2026-10-03 |
| **Demo-7** | GitHub repo and README walkthrough: clone and run without assistance | NOT STARTED | `README.md`, Gate 9 clean-clone test | 2026-10-03 |
| **Deliv-P1**| Phase 1 Deliverable: architecture diagram and working naive RAG demo with logged RAGAS baseline | NOT STARTED | `docs/ARCHITECTURE.md`, `src/prismx/__main__.py`, `results/phase1/` | 2026-10-03 |
| **Deliv-P2**| Phase 2 Deliverable: benchmarking report showing Phase 1 vs Phase 2 RAGAS & latency, repo with README | NOT STARTED | `reports/BENCHMARK_REPORT.md`, `README.md` | 2026-10-03 |
| **Bonus-1** | Reranker integration: cross-encoder reranker after retrieval (Gate 8 flag) | OUT OF SCOPE | Gate 8 optional extension module (`src/prismx/retrieve/rerank.py`) | 2026-10-03 |
| **Bonus-2** | Larger index scale: 500,000+ passages | OUT OF SCOPE | Out of scope for this task | 2026-10-03 |
| **Bonus-3** | Multi-vector retrieval: ColBERT or multi-vector passage representations | OUT OF SCOPE | Out of scope for this task | 2026-10-03 |
| **Bonus-4** | Caching layer: query result caching to reduce repeated-query latency | OUT OF SCOPE | Extension point left in `src/prismx/retrieve/service.py` | 2026-10-03 |
| **Bonus-5** | LLM-generated answers: pass retrieved context to LLM | OUT OF SCOPE | Extension point left in API schema | 2026-10-03 |
| **Bonus-6** | Evaluation dashboard: real-time RAGAS metric display | OUT OF SCOPE | Out of scope for this task | 2026-10-03 |
