"""PRISMX Request and Response Pydantic Schemas."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

class FusionParams(BaseModel):
    method: Literal["weighted", "rrf"] = "weighted"
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)

class FilterParams(BaseModel):
    category: str | list[str] | None = None
    source: str | list[str] | None = None

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    mode: Literal["dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    filters: FilterParams | None = None
    fusion: FusionParams | None = None
    rerank: bool = False

class SearchResultItem(BaseModel):
    rank: int
    passage_id: str
    text: str
    category: str | None = None
    source: str | None = None
    score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None

class LatencyBreakdown(BaseModel):
    encode: float
    dense: float
    sparse: float
    fusion: float
    fetch_text: float
    total: float

class SearchResponse(BaseModel):
    query: str
    mode: str
    fusion_used: dict[str, Any] | None = None
    filters_applied: dict[str, Any] | None = None
    index_version: int
    results: list[SearchResultItem]
    latency_ms: LatencyBreakdown

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
    categories: list[Any] | None = None
    sources: list[str] | None = None
    models: dict[str, str]
    config_hash: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
