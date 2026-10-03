# PRISMX Hybrid Fusion Specification

This document details the mathematical formulation, score normalization, tie-breaking rules, and edge-case handling for the two supported hybrid fusion strategies: **Weighted Linear Combination** and **Reciprocal Rank Fusion (RRF)**.

---

## 1. Weighted Linear Combination (`method: "weighted"`)

### Mathematical Definition
For a query $q$ and a candidate passage $d$ appearing in the candidate set retrieved from channel $c \in \{\text{dense}, \text{sparse}\}$:

$$\text{Score}_{\text{weighted}}(d) = \alpha \cdot \bar{s}_{\text{dense}}(d) + (1 - \alpha) \cdot \bar{s}_{\text{sparse}}(d)$$

where:
- $\alpha \in [0.0, 1.0]$ is the dense channel weight (production frozen default: $\alpha = 0.80$, as specified in `CONFIG.yaml`).
  > **Dataset & Tuning Provenance:** The grid search across fusion weights was conducted on the **Phase 2 Curated Partition** (150 seeded TUNE queries, `results/phase2/fusion_tuning_tune.json`), which established $\alpha = 0.80$ dense + $0.20$ BM25 sparse as the optimal trade-off (+0.0458 NDCG@5 over RRF). Informational sensitivity results on the full 100K raw MS MARCO corpus (`c100k_raw`, 500 TUNE queries across $\alpha \in \{0.6, 0.7, 0.8, 0.9, 1.0\}$) are stored in `results/c100k_raw/tune_eval_results.json` under `alpha_sensitivity_informational` (labeled *"not used for selection"*). On `c100k_raw` BENCH (N=100), hybrid search ($\alpha = 0.80$) showed no measurable retrieval quality gain over dense search ($\Delta\text{MRR@10} = -0.0083\ [-0.0466, +0.0300]$, crossing zero). ADR-018 governed the serving default mode decision based on latency, not alpha selection.
  > **Note on Z-Score Normalization:** Z-Score score normalization was considered theoretically during initial system design, but was not implemented or benchmarked (documented as considered, not run). `CONFIG.yaml`, the UI, API, and all evaluation files use $\alpha = 0.80$ min-max weighted linear fusion.
- $\bar{s}_{c}(d)$ is the min-max normalized score of passage $d$ within the channel's candidate list.
- If document $d$ was not retrieved by channel $c$, its normalized score for that channel is defined as:
  $$\bar{s}_{c}(d) = 0.0$$

### Min-Max Normalization
For a candidate set $C_c$ returned by channel $c$ with raw scores $s_c(d)$:
$$\bar{s}_c(d) = \frac{s_c(d) - \min_{d' \in C_c} s_c(d')}{\max_{d' \in C_c} s_c(d') - \min_{d' \in C_c} s_c(d')}$$

#### Edge Case: Zero Variance (All Scores Equal)
If $\max_{d'} s_c(d') == \min_{d'} s_c(d')$ (all scores in the channel are identical):
$$\bar{s}_c(d) = 1.0 \quad \forall d \in C_c$$
If $C_c = \emptyset$ (empty channel), the channel contributes $0.0$ to all documents.

---

## 2. Reciprocal Rank Fusion (`method: "rrf"`)

### Mathematical Definition
For candidate passage $d$ across retrieval channels $C$:

$$\text{Score}_{\text{RRF}}(d) = \sum_{c \in \{\text{dense}, \text{sparse}\}} \mathbb{I}(d \in C_c) \cdot \frac{1}{k_{\text{rrf}} + \text{rank}_c(d)}$$

where:
- $k_{\text{rrf}} > 0$ is a smoothing constant (default $k_{\text{rrf}} = 60$, configurable).
- $\text{rank}_c(d) \in \{1, 2, \dots, n\}$ is the 1-based rank of document $d$ in channel $c$'s results.
- $\mathbb{I}(d \in C_c)$ is $1$ if $d$ was retrieved by channel $c$, and $0$ otherwise.

---

## 3. Stable Tie-Breaking Rules

To guarantee deterministic ordering across runs, environments, and platforms (Rule R9), ties in fused scores are resolved strictly by:
1. **Primary Key:** Fused Score (descending)
2. **Secondary Key:** Dense Rank (ascending; passages retrieved by dense rank 1 before rank 2; if missing from dense, rank is treated as $\infty$)
3. **Tertiary Key:** Sparse Rank (ascending)
4. **Quaternary Key:** `passage_id` (alphabetical / numerical ascending string sort)

---

## 4. Hand-Computed Unit Test Reference Cases

### Case A: Weighted Normalization
Let $\alpha = 0.6$.
- Channel Dense:
  - docA: raw 0.90 -> normalized 1.0
  - docB: raw 0.70 -> normalized 0.0
- Channel Sparse:
  - docB: raw 10.0 -> normalized 1.0
  - docC: raw 5.0 -> normalized 0.0

Fused Scores:
- **docA:** $0.6 \times 1.0 + 0.4 \times 0.0 = 0.60$
- **docB:** $0.6 \times 0.0 + 0.4 \times 1.0 = 0.40$
- **docC:** $0.6 \times 0.0 + 0.4 \times 0.0 = 0.00$
Order: docA (0.60), docB (0.40), docC (0.00).

### Case B: RRF ($k=60$)
- Channel Dense ranks: docA (1), docB (2)
- Channel Sparse ranks: docB (1), docC (2)

Scores:
- **docA:** $1/(60+1) + 0 = 1/61 \approx 0.016393$
- **docB:** $1/(60+2) + 1/(60+1) = 1/62 + 1/61 \approx 0.016129 + 0.016393 = 0.032522$
- **docC:** $0 + 1/(60+2) = 1/62 \approx 0.016129$
Order: docB (0.032522), docA (0.016393), docC (0.016129).
