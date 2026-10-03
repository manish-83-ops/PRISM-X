"""PRISMX Pre-Retrieval Filter Converter for Qdrant."""

from __future__ import annotations

from typing import Any
from qdrant_client import models
from prismx.schemas import FilterParams

def build_qdrant_filter(filters: FilterParams | dict[str, Any] | None) -> models.Filter | None:
    """Translates category and source filter constraints into a native Qdrant pre-retrieval filter.
    
    Ensures pre-retrieval filtering at the vector database index scan level (FR-4).
    """
    if filters is None:
        return None

    if isinstance(filters, dict):
        cat = filters.get("category")
        src = filters.get("source")
    else:
        cat = filters.category
        src = filters.source

    conditions: list[models.Condition] = []

    # Category filter
    if cat:
        if isinstance(cat, list):
            if len(cat) == 1:
                conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=cat[0])))
            elif len(cat) > 1:
                conditions.append(models.FieldCondition(key="category", match=models.MatchAny(any=cat)))
        else:
            conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=cat)))

    # Source filter
    if src:
        if isinstance(src, list):
            if len(src) == 1:
                conditions.append(models.FieldCondition(key="source", match=models.MatchValue(value=src[0])))
            elif len(src) > 1:
                conditions.append(models.FieldCondition(key="source", match=models.MatchAny(any=src)))
        else:
            conditions.append(models.FieldCondition(key="source", match=models.MatchValue(value=src)))

    if not conditions:
        return None

    return models.Filter(must=conditions)
