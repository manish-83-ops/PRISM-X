# PRISMX API Contract

This document provides the complete, authoritative specification for all HTTP endpoints provided by the PRISMX backend. Frontend and UI teammates can integrate directly against this specification.

---

## 1. Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/search` | Retrieve top-$k$ passages using dense or hybrid mode with optional pre-filtering |
| `POST` | `/passages/upsert` | Add or update a single passage with atomic index version update |
| `DELETE` | `/passages/{passage_id}` | Remove a passage from Qdrant and SQLite |
| `GET` | `/health` | Healthcheck returning process liveness |
| `GET` | `/ready` | Readiness check (200 OK only after models and Qdrant are loaded and warmed up) |
| `GET` | `/meta` | System metadata, available modes, categories, point count, index version |
| `GET` | `/bench/latest` | Latest benchmark results JSON |
| `GET` | `/eval/latest` | Latest evaluation results JSON |

---

## 2. Specification

### `POST /search`

#### Request Body
```json
{
  "query": "what is machine learning",
  "mode": "hybrid",
  "top_k": 5,
  "filters": {
    "category": "science-tech",
    "source": "msmarco-passage"
  },
  "fusion": {
    "method": "weighted",
    "alpha": 0.7,
    "rrf_k": 60
  },
  "rerank": false
}
```

- `query` (string, required): 1 to 512 characters.
- `mode` (string, optional, default `"hybrid"`): `"dense"` or `"hybrid"`.
- `top_k` (integer, optional, default `5`): Range 1 to 50.
- `filters` (object, optional, nullable):
  - `category` (string, list of strings, or null).
  - `source` (string, list of strings, or null).
- `fusion` (object, optional, nullable):
  - `method` (string): `"weighted"` or `"rrf"`.
  - `alpha` (float): Required if method is `"weighted"`.
  - `rrf_k` (integer): Required if method is `"rrf"`.
- `rerank` (boolean, optional, default `false`): Applies cross-encoder reranker if enabled.

#### Response Body (200 OK)
```json
{
  "query": "what is machine learning",
  "mode": "hybrid",
  "fusion_used": {
    "method": "weighted",
    "alpha": 0.7,
    "rrf_k": 60
  },
  "filters_applied": {
    "category": ["science-tech"],
    "source": ["msmarco-passage"]
  },
  "index_version": 1,
  "results": [
    {
      "rank": 1,
      "passage_id": "349120",
      "text": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'...",
      "category": "science-tech",
      "source": "msmarco-passage",
      "score": 0.8842,
      "dense_rank": 1,
      "dense_score": 0.8415,
      "bm25_rank": 2,
      "bm25_score": 14.23
    }
  ],
  "latency_ms": {
    "encode": 6.2,
    "dense": 3.8,
    "sparse": 2.4,
    "fusion": 0.4,
    "fetch_text": 0.9,
    "total": 13.7
  }
}
```

*Notes on dense vs hybrid channel fields:*
- In `"dense"` mode, `bm25_rank` and `bm25_score` are returned as `null`.
- In `"hybrid"` mode, if a passage was discovered by only one channel, the opposite channel fields are `null`.

---

### `POST /passages/upsert`

#### Request Body
```json
{
  "passage_id": "test_doc_001",
  "text": "This is a newly ingested live update document for testing.",
  "category": "science-tech",
  "source": "manual_ingest"
}
```

#### Response Body (200 OK)
```json
{
  "status": "success",
  "passage_id": "test_doc_001",
  "index_version": 2
}
```

---

### `DELETE /passages/{passage_id}`

#### Response Body (200 OK)
```json
{
  "status": "deleted",
  "passage_id": "test_doc_001",
  "index_version": 3
}
```

---

### `GET /health` & `GET /ready`

#### `GET /health` (200 OK)
```json
{ "status": "ok" }
```

#### `GET /ready` (200 OK)
```json
{
  "status": "ready",
  "models_loaded": true,
  "qdrant_connected": true,
  "warmed_up": true
}
```

---

### Standard Error Format
All errors return consistent JSON with semantic HTTP status codes (no raw stack traces):
```json
{
  "error": "BAD_REQUEST",
  "detail": "query must be between 1 and 512 characters"
}
```
