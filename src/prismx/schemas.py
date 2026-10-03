"""PRISMX Request and Response Pydantic Schemas."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class FusionParams(BaseModel):
    method: Literal["weighted", "rrf"] = "weighted"
    alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)


class FilterParams(BaseModel):
    category: str | list[str] | None = None
    source: str | list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    mode: Literal["dense", "hybrid", "prismx", "hybrid_rerank", "hybrid+rerank"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    filters: FilterParams | None = None
    fusion: FusionParams | None = None
    rerank: bool = False
    rerank_k: int = Field(default=10, ge=1, le=100)
    search_ef: int | None = Field(default=128, ge=1, le=1000, description="HNSW search_ef (defaults to CONFIG.yaml qdrant.search_ef=128)")
    total_deadline_ms: float = Field(default=250.0, ge=10.0, le=10000.0, description="Total request deadline in ms")
    rerank_budget_ms: float | None = Field(default=200.0, ge=10.0, le=5000.0, description="Covers rerank stage micro-batch boundaries only (not total request latency)")
    deadline_ms: float | None = Field(default=None, description="Backward-compatible alias for rerank_budget_ms")
    budget_ms: float | None = Field(default=None, description="Backward-compatible alias for rerank_budget_ms")
    use_cache: bool = True
    cache: bool | None = None  # alias for use_cache

    def get_use_cache(self) -> bool:
        if self.cache is not None:
            return self.cache
        return self.use_cache

    def get_rerank_budget_ms(self) -> float:
        if self.deadline_ms is not None:
            return self.deadline_ms
        if self.budget_ms is not None:
            return self.budget_ms
        if self.rerank_budget_ms is not None:
            return self.rerank_budget_ms
        return 200.0


class SearchResultItem(BaseModel):
    rank: int
    passage_id: str
    text: str
    category: str | None = None
    source: str | None = None
    length_chars: int | None = None
    score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fused_rank: int | None = None
    fused_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None
    retrieved_by: list[str] = Field(default_factory=list)


class LatencyBreakdown(BaseModel):
    encode: float = 0.0
    dense: float = 0.0
    sparse: float = 0.0
    fusion: float = 0.0
    fetch_text: float = 0.0
    rerank: float = 0.0
    total: float = 0.0


class SearchResponse(BaseModel):
    query: str
    mode: str
    fusion_used: dict[str, Any] | None = None
    filters_applied: dict[str, Any] | None = None
    index_version: int
    results: list[SearchResultItem]
    latency_ms: LatencyBreakdown
    cache_hit: bool = False
    governor_state: str = "normal"
    stage_reached: str = "stage1_hybrid"
    candidates_scored: int = 0
    K_requested: int = 5
    per_pair_ms: float = 0.0
    effective_mode: str = "hybrid"


class UpsertRequest(BaseModel):
    passage_id: str
    text: str = Field(..., min_length=1)
    category: str | None = None
    source: str | None = "manual"


class UpsertResponse(BaseModel):
    status: str = "success"
    passage_id: str
    index_version: int


class DeleteResponse(BaseModel):
    status: str = "deleted"
    passage_id: str
    index_version: int


class MetaResponse(BaseModel):
    modes: list[str]
    fusion_defaults: dict[str, Any]
    point_count: int
    sqlite_count: int | None = None
    index_version: int
    avgdl_ref: float | None = None
    true_avgdl: float | None = None
    drift: float | None = None
    drift_warning: bool | None = None
    inconsistency_count: int | None = 0
    outbox_pending_count: int | None = 0
    consistency_probe: dict[str, Any] | None = None
    categories: list[Any] | None = None
    sources: list[str] | None = None
    models: dict[str, str]
    config_hash: str
    cache_stats: dict[str, Any] | None = None


class AnswerCitation(BaseModel):
    citation_id: int
    passage_id: str
    category: str | None = None
    source: str | None = None
    score: float


class ConfigResponse(BaseModel):
    default_mode: str
    semantic_hash: str
    serving_hash: str
    index_version: int
    corpus_size: int
    collection_name: str
    git_commit: str
    backend_status: str
    public_demo: bool = False
    rate_limits: dict[str, Any] | None = None


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["dense", "hybrid", "prismx", "hybrid_rerank", "hybrid+rerank"] = "hybrid"
    rerank_k: int = Field(default=20, ge=1, le=100)
    filters: FilterParams | None = None
    use_cache: bool = True
    model: str | None = None


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: list[AnswerCitation]
    passages: list[SearchResultItem]
    model: str
    latency_ms: dict[str, float]
    cache_hit: bool = False
    validation: dict[str, Any] | None = None
    tokens_used: int = 0


class LiveCheckRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(default_factory=list)


class LiveCheckResponse(BaseModel):
    query: str
    answer: str
    label: str = "single query, LLM-judged, indicative"
    judge_model: str
    usefulness_scores: list[dict[str, Any]]
    faithfulness: dict[str, Any]
    latency_ms: float
    tokens_used: int = 0


class ErrorResponse(BaseModel):
    error: str
    detail: str
