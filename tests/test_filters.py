"""Unit tests for Pre-Retrieval Filtering Converter (FR-4)."""

import pytest
from qdrant_client import models
from prismx.retrieve.filters import build_qdrant_filter
from prismx.schemas import FilterParams


def test_build_qdrant_filter_none():
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter({}) is None
    assert build_qdrant_filter(FilterParams()) is None


def test_build_qdrant_filter_single_category():
    # Via dict
    f1 = build_qdrant_filter({"category": "science-tech"})
    assert isinstance(f1, models.Filter)
    assert len(f1.must) == 1
    cond = f1.must[0]
    assert cond.key == "category"
    assert cond.match.value == "science-tech"

    # Via FilterParams
    f2 = build_qdrant_filter(FilterParams(category="finance-business"))
    assert isinstance(f2, models.Filter)
    assert len(f2.must) == 1
    assert f2.must[0].key == "category"
    assert f2.must[0].match.value == "finance-business"


def test_build_qdrant_filter_multiple_categories():
    cats = ["science-tech", "health-medicine"]
    f = build_qdrant_filter({"category": cats})
    assert isinstance(f, models.Filter)
    assert len(f.must) == 1
    cond = f.must[0]
    assert cond.key == "category"
    assert hasattr(cond.match, "any")
    assert cond.match.any == cats


def test_build_qdrant_filter_category_and_source():
    f = build_qdrant_filter(FilterParams(category="law-gov", source="msmarco-passage"))
    assert isinstance(f, models.Filter)
    assert len(f.must) == 2
    keys = {c.key for c in f.must}
    assert keys == {"category", "source"}
