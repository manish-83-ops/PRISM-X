# Data Management and Corpus Specification

This document details the MS MARCO passage dataset selection, schema verification, deterministic sampling strategy, split generation, sanity metrics, and critical caveats.

---

## 1. Dataset Selection and Schema Verification

### Selected HuggingFace Repositories
- **Corpus Source:** `Tevatron/msmarco-passage-corpus`
  - Stable passage IDs (`docid` string matching official MS MARCO IDs `0` to `8841822`).
  - Schema: `docid: string`, `title: string`, `text: string`.
  - Raw size: 1,066.68 MB gzipped (8,841,823 passages).
- **Query & Qrels Source:** `Tevatron/msmarco-passage` + `BeIR/msmarco-qrels`
  - Dev Queries: `Tevatron/msmarco-passage` (`dev.jsonl.gz`, 6,980 queries).
  - Dev Qrels: `BeIR/msmarco-qrels` (`dev.tsv`, 7,437 qrels across 6,980 unique queries).
  - Verification: 100% of query IDs in `dev.tsv` match `dev.jsonl.gz`. 100% of positive passage IDs exist in the corpus.

### Why Chosen
Direct official MS MARCO canonical IDs without re-indexing or synthetic ID shifts, enabling exact cross-channel parity between dense and sparse representations.

---

## 2. Deterministic Query Splits

From the 6,980 MS MARCO Dev queries that have $\ge 1$ positive qrel:
- **Seed:** `42` (deterministic random shuffle).
- **TUNE Split:** 500 queries (`split_tune.json`, SHA-256 in manifest). All hyperparameter tuning is restricted strictly to this split. Associated with 531 gold passages.
- **TEST Split:** 500 queries (`split_test.json`, SHA-256 in manifest). Frozen evaluation only, used strictly ONCE per frozen configuration. Associated with 541 gold passages.
- **BENCH Split:** First 100 queries of a fixed, seeded permutation (`seed=43`) of the TEST split. Used for latency profiling under protocol D2.
- **RAGAS Split:** The identical 100 queries as BENCH, used for RAGAS evaluation across Non-LLM and LLM-based families.
- **Partition Isolation:** $\text{TUNE} \cap \text{TEST} = \emptyset$ (0 overlap, verified). Total unique gold passages across TUNE+TEST: 1,072.

---

## 3. Corpus A ("standard-100K") Manifest and Sanity Statistics

Corpus A was constructed deterministically in a single streaming pass through `data/raw/corpus.jsonl.gz`:
1. **Passage Count:** EXACTLY 100,000 unique passages.
2. **Gold Passages:** 1,072 passages (100% of gold passages for TUNE + TEST queries).
3. **Filler Passages:** 98,928 passages sampled via lowest SHA-256 hash rank.
4. **Gold-to-Total Ratio:** 0.01072 (1.072%).
5. **Exact Duplicate Texts Removed:** 1 (gold copy preserved).
6. **Corpus File:** `data/corpus_100k.jsonl` (SHA-256: `30d5212101e6d5ee12c0f5894e0cc522fd8da526b434ab4c841c34ee1f29e8b2`).

### Passage Length Distribution
- **Characters:**
  - Min: 14
  - 25th percentile: 255.0
  - Median (p50): 301.0
  - 75th percentile: 388.0
  - 95th percentile: 598.0
  - Max: 1,299
  - Mean: 335.99
- **Words:**
  - Min: 2
  - 25th percentile: 42.0
  - Median (p50): 50.0
  - 75th percentile: 65.0
  - 95th percentile: 102.0
  - Max: 230
  - Mean: 56.29

### Query-Passage Statistics
- **Qrels Per Query:** Mean 1.072, Min 1, Max 4.
- **Verbatim Query in Gold Passage:** 2.8% of queries appear verbatim within the gold passage text.

---

## 4. Informative BM25 Baseline (bm25s) on TUNE Split

As required by Gate 1, BM25 using `bm25s` (k1=1.2, b=0.75, English stopwords) was evaluated on the 500 TUNE queries over Corpus A purely as **information**:
- **MRR@10:** 0.6386
- **Recall@20:** 0.8340

> [!WARNING]
> **Caveat Regarding 100K Sub-Corpus Evaluation:**
> A 100,000 passage sample has far fewer distractors than the full 8.8 million passage MS MARCO collection. Consequently, absolute retrieval scores (such as MRR@10 = 0.6386 and Recall@20 = 0.8340) are significantly higher than published numbers on the full MS MARCO leaderboard (which typically hover around 0.18 - 0.23 for un-reranked BM25). All comparisons in PRISMX are strictly paired and relative between Phase 1 and Phase 2 on this identical frozen 100K sub-corpus. These informational numbers were not used to modify or filter corpus content.

---

## 5. Sample Query - Gold Passage Pairs (10 Random from TUNE)

| Query ID | Query | Gold Passage ID | Gold Passage Excerpt |
| :--- | :--- | :--- | :--- |
| `578735` | what benefit did the social security act provide for people who were not of retirement age? | `7575680` | Social Security Disability (SSDI) benefits automatically convert to retirement benefits at the same rate of pay when the... |
| `1060566` | community bank bristow routing number | `7168414` | View details for routing number - 103112125 - assigned to COMMUNITY BANK in BRISTOW, OK. The ABA routing/transit number ... |
| `1007696` | when a second epsp arrives at a single synapse before the effects of the first have disappeared, what occurs? | `7251223` | When a second EPSP arrivesat a single synapse before the effects of the first one have disappeared, what occurs? Tempora... |
| `738165` | what is definition of fugue music | `7430567` | Fugue. in music, the most mature form of imitative counterpoint (see. ). The fugue is based on a short melody, or theme,... |
| `1094389` | insertion point definition | `1304256` | Insertion Point. An insertion point is the location on the screen where the next character typed will be inserted. This ... |
| `1090352` | sty causes | `2791813` | Styes are usually caused by infections of the oil glands in the eyelid. Very frequently, they are infected by bacteria, ... |
| `1089036` | vasospasms caused by what | `7088809` | Blood can also irritate and damage the normal blood vessels and cause vasospasm (constriction). This can interrupt norma... |
| `1080031` | what gao office | `7149425` | The United States Government Accountability Office (GAO) is an independent agency that investigates how the federal gove... |
| `729697` | what is chattahoochee | `7615909` | Chattahoochee, Chattahoochee River(noun) a river rising in northern Georgia and flowing southwest and south to join the ... |
| `1055197` | what is flex fuel on a jeep? | `7176630` | GM identifies its E85 ethanol flex-fuel vehicles with Flex Fuel E85 badges and yellow fuel-filler caps. Ford labels its ... |

---

## 6. Raw Query-Centric Corpus (`c100k_raw`) & Selection-Bias Audit (Gate 5.3)

### Eligibility on Sampled Queries
From the official MS MARCO v2.1 validation split (101,093 queries), a seeded permutation (`seed = 42`) was sampled to construct the uncurated candidate distribution `c100k_raw` (100,008 unique passages):
- **Total sampled queries:** 10,115
- **Queries with $\ge 1$ gold passage (`is_selected == 1`):** 5,581 (55.18%)
- **Queries with valid ADR-014 human reference answer:** 5,388 (53.27%)
- **Eligible queries ($\ge 1$ gold AND valid answer):** **5,387 (53.26%)**

> [!IMPORTANT]
> **Selection Policy:** The evaluation benchmark queries (`TUNE` 500, `BENCH` 100) are drawn strictly and disjointly from this **5,387 eligible subset**. Queries lacking labeled gold passages or lacking valid multi-token human reference answers are excluded to prevent unjudged metric confounding. Benchmark queries are drawn from this eligible subset.

### Query Type Distribution & Selection-Bias Audit

| Query Type | Sampled Count (N=10,115) | Sampled Share (%) | Eligible Count (N=5,387) | Eligible Share (%) | Absolute Difference (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DESCRIPTION** | 5,444 | 53.82% | 2,781 | 51.62% | 2.20% | Expected natural variance |
| **NUMERIC** | 2,637 | 26.07% | 1,337 | 24.82% | 1.25% | Preserved (<1.5% shift) |
| **ENTITY** | 815 | 8.06% | 471 | 8.74% | 0.68% | Preserved (<1.0% shift) |
| **PERSON** | 616 | 6.09% | 364 | 6.76% | 0.67% | Preserved (<1.0% shift) |
| **LOCATION** | 603 | 5.96% | 434 | 8.06% | 2.10% | Slightly higher answer rate |

**Takeaway:** Filtering for gold ground truth and human reference answers causes minimal distribution shift across all 5 query types ($\le 2.2\%$ deviation). The relative category rankings (DESCRIPTION > NUMERIC > ENTITY > PERSON $\approx$ LOCATION) are strictly preserved between the sampled pool and the evaluation benchmark split.

