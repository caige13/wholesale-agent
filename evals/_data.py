"""Shared eval data helpers — the dataset loader and the cart seeder.

Both eval entry points (the local ``run_eval`` runner and the LangSmith-native
``langsmith_eval``) read the same JSONL rows and rebuild a row's ``starting_cart``
the same way, so that logic lives here once rather than being copied per runner.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.domain.cart import Cart
from src.domain.models import LineItem

DATASET = Path(__file__).resolve().parent / "datasets" / "order_desk.jsonl"


def load_rows() -> list[dict]:
    """Every dataset row (one JSON object per non-blank line)."""
    lines = [ln for ln in DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def cart_from(raw: dict | None) -> Cart:
    """Rebuild a ``starting_cart`` (supplier → line dicts) into a Cart, empty if absent."""
    if not raw:
        return Cart()
    return Cart(by_supplier={s: [LineItem(**li) for li in items] for s, items in raw.items()})


def format_context(candidates) -> str:
    """Render retrieval candidates as the catalog context the faithfulness judge sees."""
    if not candidates:
        return "(no matching catalog entries)"
    return "\n".join(
        f"- {c.item.product_name} ({c.item.unit_size}, {c.item.case_pack} per case)"
        for c in candidates
    )