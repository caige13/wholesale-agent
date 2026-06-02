"""validate_rules — static-catalog business rules.

Pure function: given a line and its catalog item, return a new line with the
quantity rounded to whole cases and the catalog-derived flags raised. Only rules
that depend on *static* catalog data live here (case packs, minimums,
companions). Out-of-stock is dynamic supplier-API data and is set elsewhere.
"""

from __future__ import annotations

import math

from src.domain.models import CatalogItem, Flag, LineItem


def validate_rules(item: LineItem, catalog_item: CatalogItem) -> LineItem:
    flags = list(item.flags)
    quantity = item.quantity

    # Order given in units → round up to whole cases; flag if it wasn't exact.
    if item.unit_quantity is not None and quantity is None:
        quantity = math.ceil(item.unit_quantity / catalog_item.case_pack)
        if item.unit_quantity % catalog_item.case_pack != 0:
            _raise(flags, Flag.ROUNDED_TO_CASE_PACK)

    # No amount stated ("I want salsa cups") → ask how many rather than draft an
    # order for an unspecified quantity. Blocking, so the gate clarifies.
    if quantity is None:
        _raise(flags, Flag.MISSING_QUANTITY)

    if quantity is not None and quantity < catalog_item.min_order:
        _raise(flags, Flag.BELOW_MINIMUM)

    if catalog_item.companion_skus:
        _raise(flags, Flag.NEEDS_COMPANION)

    return item.model_copy(update={"quantity": quantity, "flags": flags})


def _raise(flags: list[Flag], flag: Flag) -> None:
    """Append a flag once (no duplicates)."""
    if flag not in flags:
        flags.append(flag)