"""validate_rules — static-catalog business rules.

Pure function: given a line and its catalog item, return a new line with the
quantity rounded to whole cases and the catalog-derived flags raised. Rules that
depend on *static* catalog data live here (case packs, minimums, companions). It
also raises ``EXCEEDS_STOCK`` when the finalized case count outruns the on-hand
count the inventory check already recorded on the line — still pure (it reads that
number off the line, it never calls the supplier). Whether stock exists *at all*
(``OUT_OF_STOCK``) is the inventory check's call, not this function's.
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

    # Over-stock: now that the quantity is whole cases, compare it against the on-hand
    # count the inventory check recorded (also whole cases). A None/0 on-hand means
    # "unknown", so it never trips here — an out-of-stock SKU is OUT_OF_STOCK instead.
    if (
        quantity is not None
        and item.quantity_on_hand is not None
        and item.quantity_on_hand > 0
        and quantity > item.quantity_on_hand
    ):
        _raise(flags, Flag.EXCEEDS_STOCK)

    if catalog_item.companion_skus:
        _raise(flags, Flag.NEEDS_COMPANION)

    return item.model_copy(update={"quantity": quantity, "flags": flags})


def _raise(flags: list[Flag], flag: Flag) -> None:
    """Append a flag once (no duplicates)."""
    if flag not in flags:
        flags.append(flag)