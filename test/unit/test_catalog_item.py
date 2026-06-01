"""CatalogItem invariants — the static-catalog data the pipeline divides by.

case_pack is a divisor in validate_rules (units → whole cases) and min_order is a
floor; a non-positive value is a malformed catalog row that should fail loudly at
construction/load time, not blow up (ZeroDivisionError) deep in the pipeline.
"""

import pytest
from pydantic import ValidationError

from src.domain.models import CatalogItem


def _kwargs(**overrides) -> dict:
    base = dict(
        sku="X",
        product_name="X",
        category="containers",
        unit_size="16oz",
        case_pack=500,
        supplier="acme-foodservice",
    )
    base.update(overrides)
    return base


def test_rejects_a_nonpositive_case_pack():
    with pytest.raises(ValidationError):
        CatalogItem(**_kwargs(case_pack=0))


def test_rejects_a_nonpositive_min_order():
    with pytest.raises(ValidationError):
        CatalogItem(**_kwargs(min_order=0))