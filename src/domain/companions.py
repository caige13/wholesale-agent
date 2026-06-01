"""Companion add-ons — the deterministic half of the upsell flow.

Two pure functions, no LLM:

* ``companion_case_count`` sizes a companion order so its unit count covers the
  parent's without going under — the math the LLM must *not* do.
* ``pending_companions`` derives the still-open offers from the cart itself (a
  flagged line whose companion isn't in the cart yet), so "did we already offer
  this?" needs no extra cross-turn state — the persisted cart is the memory.
"""

from __future__ import annotations

import math

from src.domain.cart import Cart
from src.domain.models import Companion, LineItem


def companion_case_count(parent_units: int, companion_case_pack: int) -> int:
    """Cases of a companion needed to cover ``parent_units`` without going under.

    e.g. 3 cases of 32oz deli (480/case = 1440 units) paired with a lid packed
    500/case → ``ceil(1440 / 500) = 3`` cases (1500 lids ≥ 1440). Always ≥ 1 so a
    parent with an unknown/zero unit count still gets a single case offered.
    """
    if companion_case_pack <= 0:
        return 1
    return max(1, math.ceil(parent_units / companion_case_pack))


def pending_offers(cart: Cart) -> list[tuple[LineItem, Companion]]:
    """Each still-open offer as a (parent line, companion) pair, deduped by SKU.

    A companion already in the cart is satisfied, so it drops out — which is how the
    offer stops once accepted and how re-adding a parent never re-nags. The parent
    rides along so a companion can be sized against it.
    """
    in_cart = {line.sku for line in cart.all_lines()}
    seen: set[str] = set()
    offers: list[tuple[LineItem, Companion]] = []
    for line in cart.all_lines():
        for companion in line.companions:
            if companion.sku in in_cart or companion.sku in seen:
                continue
            seen.add(companion.sku)
            offers.append((line, companion))
    return offers


def pending_companions(cart: Cart) -> list[Companion]:
    """The still-open add-on offers, companions only (for the parse prompt)."""
    return [companion for _, companion in pending_offers(cart)]