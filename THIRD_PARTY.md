# Third-Party Licenses and Terms of Use

This document catalogs the third-party models, datasets, dependencies, and external systems utilized within the PRISMX retrieval engine, along with their official license terms and source repositories.

---

## 1. Datasets

### MS MARCO (Microsoft Machine Reading Comprehension) Passage Ranking Dataset
- **Usage**: Used for offline validation, split partitioning (TUNE, TEST, BENCH, RAGAS), calibration, and evaluation benchmarks.
- **License / Terms**: **Non-Commercial Research Use Only**.
- **Licensor**: Microsoft Corporation.
- **Source Link**: [Microsoft MS MARCO Official Page](https://microsoft.github.io/msmarco/)
- **Terms Summary**: The dataset is provided strictly for non-commercial research purposes. Users may not use the dataset for commercial purposes or distribution. Enterprise production deployments must ingest and index their own proprietary enterprise documents or commercially licensed datasets.

---

## 2. Machine Learning Models

### BAAI/bge-small-en-v1.5
- **Role**: Dense bi-encoder representation (384-dimensional dense vectors, normalized cosine similarity).
- **License**: **Apache License 2.0**.
- **Author / Organization**: Beijing Academy of Artificial Intelligence (BAAI).
- **Source Link**: [Hugging Face: BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **License Link**: [Apache-2.0 on Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/LICENSE)

### cross-encoder/ms-marco-MiniLM-L-6-v2
- **Role**: Cross-encoder re-ranker for stage 3 anytime cascade re-ranking (ONNX Runtime FP32).
- **License**: **Apache License 2.0**.
- **Author / Organization**: sentence-transformers / UKP Lab.
- **Source Link**: [Hugging Face: cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- **License Link**: [Apache-2.0 on Hugging Face](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/raw/main/LICENSE)

---

## 3. Core Infrastructure and Libraries

### Qdrant Vector Search Engine
- **Role**: Dual-vector dense and sparse index with payload filtering and HNSW acceleration.
- **License**: **Apache License 2.0**.
- **Source Link**: [Qdrant GitHub Repository](https://github.com/qdrant/qdrant)
- **License Link**: [Qdrant Apache-2.0 License](https://github.com/qdrant/qdrant/blob/master/LICENSE)

### ONNX Runtime
- **Role**: High-performance cross-platform inference engine for cross-encoder reranking.
- **License**: **MIT License**.
- **Author / Organization**: Microsoft Corporation.
- **Source Link**: [ONNX Runtime GitHub Repository](https://github.com/microsoft/onnxruntime)
- **License Link**: [ONNX Runtime MIT License](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)

### FastAPI
- **Role**: High-performance ASGI web framework for serving HTTP endpoints.
- **License**: **MIT License**.
- **Author / Organization**: Sebastián Ramírez.
- **Source Link**: [FastAPI GitHub Repository](https://github.com/tiangolo/fastapi)
- **License Link**: [FastAPI MIT License](https://github.com/tiangolo/fastapi/blob/master/LICENSE)

### BM25S
- **Role**: Ultra-fast sparse BM25 indexing and lexical retrieval.
- **License**: **MIT License**.
- **Author / Organization**: Xing Han Lù.
- **Source Link**: [BM25S GitHub Repository](https://github.com/xhluca/bm25s)
- **License Link**: [BM25S MIT License](https://github.com/xhluca/bm25s/blob/main/LICENSE)

### Sentence-Transformers
- **Role**: Embedding extraction framework and model utilities.
- **License**: **Apache License 2.0**.
- **Source Link**: [Sentence-Transformers GitHub Repository](https://github.com/UKPLab/sentence-transformers)
- **License Link**: [Sentence-Transformers Apache-2.0 License](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)

### PyTorch
- **Role**: Tensor library and model architecture definitions.
- **License**: **Modified BSD (BSD-3-Clause) License**.
- **Author / Organization**: Meta Platforms, Inc. / Linux Foundation.
- **Source Link**: [PyTorch GitHub Repository](https://github.com/pytorch/pytorch)
- **License Link**: [PyTorch BSD License](https://github.com/pytorch/pytorch/blob/main/LICENSE)

---

## 4. LLM API Integration

### Groq Cloud API
- **Role**: High-speed inference for RAG generation (`/answer` endpoint) using `llama-3.1-8b-instant`.
- **Terms**: Governed by Groq Cloud Terms of Service.
- **Source Link**: [Groq Cloud Terms of Service](https://groq.com/terms-of-use/)
