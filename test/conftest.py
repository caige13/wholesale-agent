"""Unit/integration test fixtures — lightweight domain-object factories.

These keep the test-first specs readable: each test builds exactly the
``CatalogItem`` / ``ResolutionCandidate`` it needs without repeating Pydantic
boilerplate. A ``make_line_item`` factory returns alongside the order-pipeline
models when the UX-driven slice reintroduces them.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.domain.models import CatalogItem, ResolutionCandidate

SUPPLIER = "acme-foodservice"


@pytest.fixture
def make_catalog_item() -> Callable[..., CatalogItem]:
    """Factory for a CatalogItem with sensible defaults; override per test."""

    def _make(
        sku: str = "DELI-16",
        product_name: str = "16oz Deli Container",
        aliases: list[str] | None = None,
        category: str = "containers",
        unit_size: str = "16oz",
        case_pack: int = 500,
        min_order: int = 1,
        companion_skus: list[str] | None = None,
        supplier: str = SUPPLIER,
    ) -> CatalogItem:
        return CatalogItem(
            sku=sku,
            product_name=product_name,
            aliases=aliases or [],
            category=category,
            unit_size=unit_size,
            case_pack=case_pack,
            min_order=min_order,
            companion_skus=companion_skus or [],
            supplier=supplier,
        )

    return _make


@pytest.fixture
def make_candidate(make_catalog_item) -> Callable[..., ResolutionCandidate]:
    """Factory for a scored ResolutionCandidate (the retriever's output shape)."""

    def _make(score: float = 0.9, **item_kwargs) -> ResolutionCandidate:
        return ResolutionCandidate(item=make_catalog_item(**item_kwargs), score=score)

    return _make