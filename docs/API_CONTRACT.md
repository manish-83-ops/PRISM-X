# PRISMX API Contract (Gate 4A)

This document provides the complete, authoritative specification for all HTTP endpoints provided by the PRISMX backend. Frontend and UI applications can integrate directly against this specification with full backward compatibility.

---

## 1. Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/search` | Retrieve top-$k$ passages using dense, hybrid, or hybrid+rerank mode with optional pre-filtering and caching |
| `POST` | `/answer` | End-to-end RAG answer synthesis grounded strictly in retrieved passages with citation markers |
| `POST` | `/passages/upsert` | Add or update a single passage with atomic index version update and cache invalidation |
| `DELETE` | `/passages/{passage_id}` | Remove a passage from Qdrant and SQLite with automatic cache invalidation |
| `POST` | `/cache/invalidate` | Manually clear all entries in the query result cache |
| `GET` | `/health` | Healthcheck returning process liveness |
| `GET` | `/ready` | Readiness check (200 OK only after models, reranker, and Qdrant are loaded and warmed up) |
| `GET` | `/meta` | System metadata, available modes, categories, point count, index version, cache statistics |
| `GET` | `/bench/latest` | Latest benchmark results JSON |
| `GET` | `/eval/latest` | Latest evaluation results JSON |

---

## 2. Specification

### `POST /search`

#### Request Body
```json
{
  "query": "what is machine learning",
  "mode": "hybrid_rerank",
  "top_k": 5,
  "rerank_k": 30,
  "use_cache": true,
  "filters": {
    "category": "science-tech",
    "source": "msmarco-passage"
  },
  "fusion": {
    "method": "weighted",
    "alpha": 0.8,
    "rrf_k": 60
  },
  "rerank": false
}
```

- `query` (string, required): 1 to 512 characters.
- `mode` (string, optional, default `"hybrid"`): Supported modes: `"dense"`, `"hybrid"`, `"hybrid_rerank"`, `"hybrid+rerank"`.
- `top_k` (integer, optional, default `5`): Range 1 to 50 (number of final passages returned).
- `rerank_k` (integer, optional, default `10`): Range 1 to 100 (candidate depth evaluated by cross-encoder in rerank modes; Gate 4B frozen default is 10).
- `rerank_budget_ms` (float, optional, default `200.0`, alias `deadline_ms`): Range 10.0 to 5000.0 (wall-clock latency budget allocated strictly to the cross-encoder reranking stage).
- `total_deadline_ms` (float, optional, default `250.0`): Range 10.0 to 10000.0 (end-to-end request latency ceiling). The cross-encoder stage receives `min(rerank_budget_ms, total_deadline_ms - elapsed_pre_rerank - 10.0ms safety)`.
- `deadline_ms` (float, optional, default `200.0`): Backward-compatible alias for `rerank_budget_ms`.
- `use_cache` (boolean, optional, default `true`): Toggle query result cache lookup and population (alias: `cache`).
- `filters` (object, optional, nullable):
  - `category` (string, list of strings, or null).
  - `source` (string, list of strings, or null).
- `fusion` (object, optional, nullable):
  - `method` (string): `"weighted"` or `"rrf"`.
  - `alpha` (float): Weight for dense channel if method is `"weighted"` (default: 0.8).
  - `rrf_k` (integer): Parameter if method is `"rrf"` (default: 60).
- `rerank` (boolean, optional, default `false`): Enables cross-encoder reranking if `mode` is `"hybrid"`.

#### Response Body (200 OK)
```json
{
  "query": "what is machine learning",
  "mode": "hybrid_rerank",
  "fusion_used": {
    "method": "weighted",
    "alpha": 0.8,
    "rrf_k": 60
  },
  "filters_applied": {
    "category": ["science-tech"],
    "source": ["msmarco-passage"]
  },
  "index_version": 1,
  "cache_hit": false,
  "governor_state": "normal",
  "results": [
    {
      "rank": 1,
      "passage_id": "349120",
      "text": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'...",
      "category": "science-tech",
      "source": "msmarco-passage",
      "length_chars": 348,
      "score": 8.4125,
      "dense_rank": 1,
      "dense_score": 0.8415,
      "bm25_rank": 2,
      "bm25_score": 14.23,
      "sparse_rank": 2,
      "sparse_score": 14.23,
      "fused_rank": 1,
      "fused_score": 0.8842,
      "rerank_rank": 1,
      "rerank_score": 8.4125,
      "retrieved_by": ["dense", "sparse"]
    }
  ],
  "latency_ms": {
    "encode": 6.2,
    "dense": 3.8,
    "sparse": 2.4,
    "fusion": 0.4,
    "fetch_text": 0.9,
    "rerank": 42.1,
    "total": 55.8
  }
}
```

*Evidence Fields Explanation:*
- `length_chars`: Character length of hydrated passage text.
- `score`: Final ranking score (rerank logit in rerank mode, fused score in hybrid mode, cosine similarity in dense mode).
- `fused_rank`, `fused_score`: Preserved first-stage fusion rank and score before reranking.
- `rerank_rank`, `rerank_score`: Cross-encoder ranking and logit score.
- `retrieved_by`: List of channels discovering the passage (`["dense"]`, `["sparse"]`, or `["dense", "sparse"]`).
- `cache_hit`: `true` if served directly from in-memory LRU cache in sub-millisecond time.
- `governor_state`: State of deadline governor (`"normal"` in 4A).

---

### `POST /answer`

#### Request Body
```json
{
  "query": "what is hypertension and its primary risks",
  "top_k": 5,
  "mode": "hybrid_rerank",
  "rerank_k": 30,
  "use_cache": true,
  "filters": null
}
```

#### Response Body (200 OK)
```json
{
  "query": "what is hypertension and its primary risks",
  "answer": "Hypertension (high blood pressure) is a condition where arterial pressure is chronically elevated [1]. Primary complications include heart attacks, strokes, and renal failure [2].",
  "citations": [
    {
      "citation_id": 1,
      "passage_id": "1110331",
      "category": "symptoms-pain",
      "source": "msmarco-passage",
      "score": 7.9124
    },
    {
      "citation_id": 2,
      "passage_id": "8731150",
      "category": "symptoms-pain",
      "source": "msmarco-passage",
      "score": 6.5412
    }
  ],
  "passages": [...],
  "model": "llama-3.3-70b-versatile",
  "latency_ms": {
    "retrieval": 65.2,
    "llm": 420.5,
    "total": 485.7
  },
  "cache_hit": false
}
```

---

### `POST /passages/upsert`
Synchronously upserts passage into SQLite and Qdrant, increments `index_version`, and invalidates the query result cache.

### `DELETE /passages/{passage_id}`
Synchronously deletes passage from SQLite and Qdrant, increments `index_version`, and invalidates the query result cache.

### `POST /cache/invalidate`
Clears all entries in the in-memory query cache. Returns `{"status": "cleared", "entries_cleared": N}`.
