# API Notes and Verified Signatures

This document records the exact signatures, verified behaviors, and differences from expected APIs for all libraries used in PRISMX. Every entry is verified via inspect/runtime checks on the installed version.

---

## 1. Qdrant Client (`qdrant-client`)
- **Version:** `1.19.1`
- **Verification Date:** 2026-10-03
- **Server Version:** `qdrant 1.19.1` (Official native Windows MSVC binary `bin/qdrant.exe`)

### `QdrantClient.__init__`
- **Signature:**
  ```python
  QdrantClient(
      url: str | None = None,
      port: int | None = 6333,
      grpc_port: int = 6334,
      prefer_grpc: bool = False,
      https: bool | None = None,
      api_key: str | None = None,
      timeout: int | None = None,
      host: str | None = None,
      path: str | None = None,
      ...
  )
  ```
- **Verified Behavior:** Connecting with `url='http://127.0.0.1:6333'`, `grpc_port=6334`, `prefer_grpc=True` connects over gRPC to the native Qdrant server.

### Search API: `query_points` vs `search`
- **Important Finding:** `QdrantClient.search` has been deprecated and completely removed in `qdrant-client` 1.19.1! Attempting to call `client.search` raises `AttributeError: type object 'QdrantClient' has no attribute 'search'`.
- **Supported Method:** `client.query_points`
- **Signature:**
  ```python
  query_points(
      collection_name: str,
      query: Union[List[float], models.SparseVector, ...],
      using: str | None = None,
      prefetch: Union[models.Prefetch, List[models.Prefetch], None] = None,
      query_filter: Union[models.Filter, None] = None,
      search_params: Union[models.SearchParams, None] = None,
      limit: int = 10,
      offset: int | None = None,
      with_payload: Union[bool, List[str], ...] = True,
      with_vectors: Union[bool, List[str]] = False,
      score_threshold: float | None = None,
      timeout: int | None = None,
  ) -> models.QueryResponse
  ```
- **Return Type:** `QueryResponse(points=[ScoredPoint(id=..., version=..., score=..., payload=..., vector=...), ...])`.
- **Sparse Vector Query:** `query=models.SparseVector(indices=[...], values=[...]), using='bm25'`.
- **Dense Vector Query:** `query=[float, ...], using='dense'`.

### Sparse Vector Configuration & IDF Modifier
- **Enum Name:** `models.Modifier.IDF` (in Python enum, value is string `'idf'`). Note: `models.Modifier.idf` raises `AttributeError: idf. Did you mean: 'IDF'?`.
- **Signature:**
  ```python
  models.SparseVectorParams(
      index: Optional[models.SparseIndexParams] = None,
      modifier: Optional[models.Modifier] = None  # e.g. models.Modifier.IDF
  )
  ```
- **Collection Creation Example:**
  ```python
  client.create_collection(
      collection_name="prismx_corpus",
      vectors_config={
          "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
      },
      sparse_vectors_config={
          "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
      }
  )
  ```

### Payload Index Creation
- **Signature:**
  ```python
  create_payload_index(
      collection_name: str,
      field_name: str,
      field_schema: Union[models.PayloadSchemaType, ...],
      wait: bool = True
  ) -> models.UpdateResult
  ```
- **Verified Value:** `models.PayloadSchemaType.KEYWORD` for `category` and `source`. With `wait=True`, returns `status=<UpdateStatus.COMPLETED: 'completed'>`.

### Upsert and Delete Write-Wait Semantics
- **Upsert Signature:**
  ```python
  upsert(
      collection_name: str,
      points: Union[models.Batch, Sequence[models.PointStruct]],
      wait: bool = True
  ) -> models.UpdateResult
  ```
  Points require `id: int | str | UUID`, `vector: dict[str, ...]` (e.g. `{"dense": [...], "bm25": models.SparseVector(...)}`), and `payload: dict`.
- **Delete Signature:**
  ```python
  delete(
      collection_name: str,
      points_selector: Union[List[int | str], models.PointIdsList, models.Filter],
      wait: bool = True
  ) -> models.UpdateResult
  ```
  Passing `wait=True` guarantees that the write or deletion is immediately applied and visible to subsequent queries.

---

## 2. Sentence Transformers (`sentence-transformers`)
- **Version:** `5.7.0`
- **Verification Date:** 2026-10-03
- **Model:** `BAAI/bge-small-en-v1.5`
- **Key Methods:**
  ```python
  SentenceTransformer(model_name_or_path: str, device: str = None)
  encode(
      sentences: Union[str, List[str]],
      batch_size: int = 32,
      show_progress_bar: bool = False,
      normalize_embeddings: bool = True,
      convert_to_numpy: bool = True
  ) -> np.ndarray
  ```

---

## 3. BM25s (`bm25s`)
- **Version:** `0.3.11`
- **Verification Date:** 2026-10-03
- **Usage:**
  ```python
  import bm25s
  retriever = bm25s.BM25(k1=1.2, b=0.75)
  corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=None)
  retriever.index(corpus_tokens)
  query_tokens = bm25s.tokenize(queries, stopwords="en", stemmer=None)
  results, scores = retriever.retrieve(query_tokens, k=10)
  ```

---

## 4. Groq Client (`groq`)
- **Version:** `0.37.1`
- **Verification Date:** 2026-10-03
- **Key Methods:**
  ```python
  from groq import Groq
  client = Groq(api_key=os.environ["GROQ_API_KEY"])
  chat_completion = client.chat.completions.create(
      messages=[{"role": "user", "content": prompt}],
      model="llama-3.3-70b-versatile",  # Or available free-tier model
      temperature=0.0
  )
  ```
- **Error Handling:** Handles `groq.RateLimitError` (HTTP 429) with exponential backoff and jitter.

---

## 5. RAGAS Metrics
- **Version:** `0.4.3`
- **Verification Date:** 2026-10-03
- **Input Requirements:**
  - Context Precision: Requires `question` (or `user_input`), `contexts` (or `retrieved_contexts`), `reference` (ground truth string).
  - Context Recall: Requires `question`, `contexts`, `reference`.
  - Non-LLM Family: Non-LLM context precision checks the position of ground truth context in retrieved contexts; Non-LLM context recall checks whether ground truth context was retrieved.
  - LLM-Based Family: Groq judge prompt evaluates whether the retrieved contexts provide the necessary facts to answer the question compared to the reference context.
