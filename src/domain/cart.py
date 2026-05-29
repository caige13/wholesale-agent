"""Cart — the order aggregate. Owns its mutation invariants and reads.

Keeping all cart behavior here (never in the LLM, never in UI callbacks) is what
keeps the UI a swappable thin mirror. ``apply`` is pure: it returns a NEW Cart
and never mutates the receiver, so the previous turn's state stays intact for the
UI / a future checkpointer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import CartOp, CartOpKind, LineItem


class Cart(BaseModel):
    """The running draft order, keyed by supplier. Persists across turns; the UI
    mirrors it. Supplier-keyed from day 1 (Stage-2 multi-supplier seam).
    """

    by_supplier: dict[str, list[LineItem]] = Field(default_factory=dict)

    def apply(self, ops: list[CartOp]) -> Cart:
        """Apply ops and return a new Cart. Lines match by (supplier, sku)."""
        by_supplier = {supplier: list(items) for supplier, items in self.by_supplier.items()}
        for op in ops:
            _apply_one(by_supplier, op)
        # Drop suppliers whose last line was removed, so the UI shows no empty group.
        return Cart(by_supplier={s: items for s, items in by_supplier.items() if items})

    def all_lines(self) -> list[LineItem]:
        """Every line across suppliers, flattened — for rendering and summaries."""
        return [item for items in self.by_supplier.values() for item in items]

    def is_empty(self) -> bool:
        return not self.all_lines()


def _apply_one(by_supplier: dict[str, list[LineItem]], op: CartOp) -> None:
    item = op.item
    items = by_supplier.setdefault(item.supplier, [])
    idx = next((i for i, li in enumerate(items) if li.sku == item.sku), None)

    if op.op is CartOpKind.ADD:
        if idx is None:
            items.append(item)
        else:
            merged = (items[idx].quantity or 0) + (item.quantity or 0)
            items[idx] = items[idx].model_copy(update={"quantity": merged})
    elif op.op is CartOpKind.SET_QUANTITY:
        if idx is None:
            items.append(item)
        else:
            items[idx] = items[idx].model_copy(update={"quantity": item.quantity})
    elif op.op is CartOpKind.REMOVE:
        if idx is not None:
            items.pop(idx)