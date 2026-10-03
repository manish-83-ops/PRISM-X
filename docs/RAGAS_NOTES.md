# RAGAS Context Precision & Recall Source Audit

**Date:** 2026-10-04  
**RAGAS Version:** `0.4.3` (Installed at `C:\Users\manis\AppData\Local\Programs\Python\Python311\Lib\site-packages\ragas`)  
**Reference Sources:**
- `ragas/metrics/_context_precision.py` (Lines 28–171)
- `ragas/metrics/_context_recall.py` (Lines 29–158)

---

## 1. Context Precision (`LLMContextPrecisionWithReference`)

### Verdict Independence
**Is the CP verdict computed per context independently?**  
**YES, 100% independently.**

As shown in `_ascore()` (`_context_precision.py` lines 147–162):
```python
responses = []
for context in retrieved_contexts:
    verdicts: t.List[Verification] = await self.context_precision_prompt.generate_multiple(
        data=QAC(
            question=user_input,
            context=context,
            answer=reference,
        ),
        llm=self.llm,
        callbacks=callbacks,
    )
    responses.append([result.model_dump() for result in verdicts])
```
Each retrieved context $c_i$ is submitted in isolation to the LLM judge alongside `user_input` and `reference`. There is **zero cross-context visibility**; the prompt contains exactly one passage at a time. Therefore, any binary relevance verdict $v_i \in \{0, 1\}$ for a passage $c_i$ depends **solely** on $(qid, \text{passage\_id}, \text{judge\_model}, \text{prompt\_hash}, \text{ragas\_version})$. If the same passage is retrieved in rank 1 for Dense and rank 4 for Rerank, its verdict $v_i$ is identical.

### Prompt Specification
Prompt class: `ContextPrecisionPrompt(PydanticPrompt[QAC, Verification])`  
Instruction:
> *"Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output."*

Output schema:
```json
{"reason": "<str>", "verdict": 0 | 1}
```

### Exact Aggregation Formula & Denominator
In `_calculate_average_precision(verifications)` (`_context_precision.py` lines 113–131):
```python
verdict_list = [1 if ver.verdict else 0 for ver in verifications]
denominator = sum(verdict_list) + 1e-10
numerator = sum(
    [
        (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
        for i in range(len(verdict_list))
    ]
)
score = numerator / denominator
```

Mathematically:
$$\text{Precision@}(k) = \frac{\sum_{j=1}^{k} v_j}{k}$$
$$\text{Numerator} = \sum_{k=1}^{K} \Big( \text{Precision@}(k) \times v_k \Big)$$
$$\text{Denominator} = \sum_{k=1}^{K} v_k + 10^{-10}$$
$$\text{Context Precision} = \frac{\sum_{k=1}^{K} \left(\frac{\sum_{j=1}^k v_j}{k} \cdot v_k\right)}{\sum_{k=1}^{K} v_k + 10^{-10}}$$

**Crucial Denominator Clarification:**
- The denominator is $\sum_{k=1}^K v_k$ (the total number of **relevant** contexts retrieved in top-$K$), **NOT** $K$ (the total number of retrieved contexts).
- If no contexts are relevant ($\sum v_k = 0$), $\text{numerator} = 0$, so $\text{CP} = 0.0$.
- If exactly 1 context is relevant and it is at rank 1: $\text{Numerator} = \frac{1}{1} \cdot 1 = 1$, $\text{Denominator} = 1$, $\text{CP} = 1.0$.
- If exactly 1 context is relevant and it is at rank 3: $\text{Numerator} = \frac{1}{3} \cdot 1 = 0.3333$, $\text{Denominator} = 1$, $\text{CP} = 0.3333$.
- If contexts at rank 1 and rank 2 are both relevant: $\text{Numerator} = 1.0 + 1.0 = 2.0$, $\text{Denominator} = 2$, $\text{CP} = 1.0$.

---

## 2. Context Recall (`LLMContextRecall`)

### Context Presentation
**Is the CR input concatenated contexts?**  
**YES, strictly concatenated.**

As shown in `_ascore()` (`_context_recall.py` lines 135–145):
```python
classifications_list = await self.context_recall_prompt.generate_multiple(
    data=QCA(
        question=row["user_input"],
        context="\n".join(row["retrieved_contexts"]),
        answer=row["reference"],
    ),
    llm=self.llm,
    callbacks=callbacks,
)
```
The retrieved contexts are joined into a single text block (`"\n".join(contexts)`). The LLM is asked to decompose the `reference` answer into sentences/statements and classify whether each assertion is supported by the combined text block.

### Aggregation Formula
```python
response = [1 if item.attributed else 0 for item in responses]
denom = len(response)
numerator = sum(response)
score = numerator / denom if denom > 0 else np.nan
```
$$\text{Context Recall} = \frac{\text{Number of Reference Statements Attributed to Concatenated Contexts}}{\text{Total Statements in Reference}}$$

---

## 3. Implications for PRISMX Architecture

1. **Persistent Verdict Ledger:**
   Since Context Precision evaluates each passage independently against $(qid, \text{passage\_text}, \text{reference})$, verdicts can be stored in a deterministic ledger keyed by `(qid, passage_id, judge, prompt_hash, ragas_version)`.
   Across Dense, Hybrid, and Rerank, identical retrieved passages share verdicts with **0 token cost**.
2. **Context Recall Caching:**
   Since Context Recall evaluates the set of passages jointly, it is invariant to ranking permutations of the same retrieved set. It can be cached by `(qid, sorted_passage_ids)`.
3. **Explaining the Metric Drop / Invariance:**
   Because the CP denominator is $\sum v_k$, if a cross-encoder brings a highly informative sibling passage into rank 1 that the judge deems non-essential relative to the strict human reference answer, Precision@1 drops to 0, pulling the entire average precision down even if gold is at rank 2.
