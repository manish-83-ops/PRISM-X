# PRISMX: High-Precision Hybrid RAG Vector Database & Retrieval Engine

PRISMX is a high-performance vector database and information retrieval pipeline designed for precision retrieval in enterprise RAG systems, eliminating LLM hallucinations through hybrid search (dense embeddings + BM25 sparse vectors with IDF modifiers) and pre-retrieval metadata filtering.

Developed for the Adrosonic Build Challenge (`docs/problem_statement.pdf`).

---

## Architecture Overview

```
                      [ User Query ]
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      [ Dense Encoder ]          [ BM25 Tokenizer ]
      (BGE-small-en-v1.5)        (Sparse Term Freq)
               │                         │
               ▼                         ▼
      [ Qdrant Dense HNSW ]      [ Qdrant Sparse + IDF ]
      (Filtered Candidates)      (Filtered Candidates)
               │                         │
               └────────────┬────────────┘
                            ▼
                  [ Hybrid Fusion ]
                  (Weighted / RRF)
                            │
                            ▼
                  [ SQLite Text Store ]
                  (Top-k Hydration)
                            │
                            ▼
                  [ Top-K Passages ]
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.10 or 3.11
- Windows, Linux, or macOS
- Qdrant Vector Database (native binary or Docker)

### 2. Environment Setup

#### PowerShell (Windows):
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package in development mode
pip install -e .
```

#### Bash (Linux/macOS):
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in development mode
pip install -e .
```

### 3. Launch Qdrant Server

#### Native Binary (Windows):
```powershell
.\bin\qdrant.exe --config-path config/qdrant.yaml
```

#### Docker:
```bash
docker compose up -d
```

### 4. Running the API and CLI

#### Start API Server:
```bash
python -m prismx server --port 8000
```

#### Execute CLI Search:
```bash
python -m prismx search --mode hybrid --query "what causes high blood pressure"
```

---

## Project Structure
```
prismx/
├── README.md               # Master documentation and quickstart
├── CONFIG.yaml             # System configuration
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment variables template
├── docker-compose.yml      # Qdrant container definition
├── pyproject.toml          # Build configuration
├── docs/                   # Full documentation & specifications
│   ├── problem_statement.pdf
│   ├── 00_REQUIREMENTS_MATRIX.md
│   ├── LOG.md
│   ├── DECISIONS.md
│   ├── ISSUES.md
│   ├── API_NOTES.md
│   ├── API_CONTRACT.md
│   ├── ARCHITECTURE.md
│   ├── FUSION.md
│   └── DATA.md
├── src/prismx/             # Core source package
│   ├── data/               # Ingestion, corpus builder, splits
│   ├── index/              # Dense encoder, lexical, Qdrant store, SQLite store
│   ├── retrieve/           # Dense, hybrid, fusion, filters, service
│   ├── api/                # FastAPI application, routes, middleware
│   └── eval/               # Metrics, bootstrap CI, benchmarks, RAGAS
├── scripts/                # Utility scripts (env, secrets, reports)
├── tests/                  # Unit and integration test suites
├── experiments/            # Experiment manifests and logs
├── results/                # Evaluation results and artifacts
└── data/manifests/         # Committed data manifests
```
