# Data Management and Corpus Specification

This document details the MS MARCO passage dataset selection, schema verification, deterministic sampling strategy, split generation, and critical caveats.

---

## 1. Dataset Selection and Schema Verification

### Selected HuggingFace Repositories
- **Corpus Source:** `Tevatron/msmarco-passage-corpus`
  - Stable passage IDs (`docid` or `passage_id` mapped to integer or string).
  - Schema: `docid: string`, `title: string`, `text: string`.
- **Query & Qrels Source:** `Tevatron/msmarco-passage`
  - Clean dev split with queries (`query_id: string`, `query: string`) and qrels mapping `query_id -> positive_passages: list[string]`.

### Verification Protocol
Before building the full corpus, sampled qrel passage IDs are verified to be 100% present in the corpus source.

---

## 2. Deterministic Splits

From the MS MARCO Dev queries that have $\ge 1$ positive qrel passage:
1. **TUNE Split (500 queries):** Used exclusively for all hyperparameter tuning (dense instruction prefix, BM25 $k_1$ and $b$, stemming, hybrid fusion $\alpha$, RRF $k_{\text{rrf}}$, and candidate depth $n$).
2. **TEST Split (500 queries):** Used **EXACTLY ONCE** for the final reported evaluation of the frozen configurations (Phase 1 baseline, Phase 2 hybrid, optional reranker).
3. **BENCH Split (100 queries):** The first 100 queries of a fixed, seeded permutation of TEST, used for client-side latency profiling under protocol D2.
4. **RAGAS Split (100 queries):** The identical 100 queries as BENCH, used for RAGAS evaluation across Non-LLM and LLM-based families.

**Partition Isolation Guarantee:** TUNE and TEST splits are strictly disjoint ($\text{TUNE} \cap \text{TEST} = \emptyset$). All query ID manifests are persisted to `data/manifests/`.

---

## 3. Corpus A ("standard-100K") Construction

Corpus A consists of **EXACTLY 100,000 unique passages**, created deterministically:
1. **Gold Retention:** All gold passages associated with the 1,000 TUNE + TEST queries are unconditionally included.
2. **Deterministic Filler Sampling:** Non-gold passages from the corpus are ranked by a stable seeded SHA-256 hash of their ID (`int(sha256(f"{seed}_{docid}"))`). The lowest-hash passages fill the remaining quota.
3. **Deduplication:** Exact text duplicate passages are removed (retaining the gold instance if a collision occurs).
4. **Final Exact Trim:** Trimmed to exactly 100,000 total passages.
5. **Leakage Prevention:** No field or payload ever stores whether a passage is gold, negative, or filler.

---

## 4. Derived Metadata Categories

MS MARCO does not contain native category labels. To support FR-4 (pre-retrieval metadata filtering):
- Embeddings are clustered into $k=15$ clusters using seeded `MiniBatchKMeans` (seed 42).
- Each cluster is labeled with its top c-TF-IDF keyword terms.
- A clean slug (e.g. `tech-hardware`, `health-medical`, `finance-econ`) is assigned to the `category` payload field.
- The `source` field is populated with `"msmarco-passage"`.

---

## 5. Critical Caveat on Retrieval Scores

> [!WARNING]
> **Caveat Regarding Evaluation on a 100,000 Passage Sub-Corpus:**
> Standard MS MARCO benchmarks evaluate over the full 8.8 million passage collection. The 100,000 passage index mandated by the challenge contains significantly fewer distractors than the full collection. Consequently, absolute retrieval metrics (Hit@1, MRR@10, NDCG@10, Recall@20) will naturally be higher than published full-corpus MS MARCO leaderboards. This sub-corpus is used strictly for relative comparison between Phase 1 (dense baseline) and Phase 2 (hybrid + metadata filtering).
