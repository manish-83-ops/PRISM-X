# PRISMX Business Case and Production Viability Analysis

## Executive Summary
PRISMX is an enterprise-grade hybrid retrieval and augmented generation engine designed to achieve high precision, sub-100ms retrieval latency, and anytime fallback guarantees on 100K+ passage corpora.

---

## 1. System Architecture and Commercial Viability
The PRISMX retrieval engine is built upon modular, permissively licensed open-source components:
- **Serving & Orchestration Codebase**: Licensed under [Apache License 2.0](../LICENSE).
- **Dense Embedding Model (`BAAI/bge-small-en-v1.5`)**: Licensed under [Apache License 2.0](https://huggingface.co/BAAI/bge-small-en-v1.5).
- **Cross-Encoder Model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**: Licensed under [Apache License 2.0](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2).
- **Vector Database (`Qdrant`)**: Licensed under [Apache License 2.0](https://github.com/qdrant/qdrant).
- **Inference Runtime (`ONNX Runtime`)**: Licensed under [MIT License](https://github.com/microsoft/onnxruntime).

All core software components, algorithms, APIs, fusion mechanisms, and cascade governors are free of restrictive copyleft licenses, making PRISMX well-suited for proprietary and enterprise deployments.

---

## 2. Dataset Licensing Terms and Commercial Boundary Limits

> [!IMPORTANT]
> **Dataset Terms Limit Notice**:
> The development, parameter tuning, ablation studies, and evaluation benchmarks conducted in this repository utilize Microsoft's **MS MARCO Passage Ranking Dataset**.
> 
> According to Microsoft's official license terms:
> - **Source Link**: [Microsoft MS MARCO Official Page](https://microsoft.github.io/msmarco/)
> - **License Classification**: Non-Commercial Research Use Only.
> - **Commercial Limit**: The MS MARCO dataset and its derived passage collections cannot be deployed or distributed directly in commercial production offerings without a separate commercial agreement from Microsoft.

### Path to Commercial Production Deployment
For enterprise or commercial applications:
1. **Model Weights Permissibility**: The pre-trained weights for `bge-small-en-v1.5` and `ms-marco-MiniLM-L-6-v2` are published under permissive Apache-2.0 licenses.
2. **Corpus Ingestion Separation**: Enterprise production environments must populate the vector database with the enterprise's own internal documentation, proprietary corpora, or commercially licensed datasets via the PRISMX Ingestion Pipeline (`prismx.ingest`).
3. **Evaluation Protocol**: Commercial evaluations should employ domain-specific ground-truth sets derived from internal enterprise queries.

---

## 3. Operational ROI & Total Cost of Ownership (TCO)
1. **Hardware Efficiency**: By executing FP32 ONNX Runtime inference on standard multi-core CPUs without requiring dedicated GPU infrastructure, PRISMX reduces serving infrastructure costs by an estimated 70–80% compared to heavy GPU-bound rerankers.
2. **Anytime Cascade Governor**: SLA breaches are mechanically prevented through real-time deadline monitoring and micro-batch truncation, eliminating tail-latency cascades that degrade downstream user experience.
3. **Decoupled Architecture**: Storage of full document text in SQLite and embedding vectors in Qdrant ensures that index growth does not bloat expensive in-memory vector storage.
