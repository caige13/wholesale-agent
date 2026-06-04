"""Direct cart edits from the panel (remove / change quantity).

The user can edit the cart without going through the agent, but mutation still
flows through the domain's one mutation point (``Cart.apply``): the UI only
constructs the declarative op — the same shape the graph emits (spec §6/§11).
"""

from __future__ import annotations

from src.domain.cart import Cart
from src.domain.models import CartOp, CartOpKind, LineItem


def remove_line(cart: Cart, supplier: str, sku: str) -> Cart:
    """Return a new cart with the (supplier, sku) line removed."""
    op = CartOp(op=CartOpKind.REMOVE, item=LineItem(sku=sku, supplier=supplier))
    return cart.apply([op])


def set_line_quantity(cart: Cart, supplier: str, sku: str, quantity: int) -> Cart:
    """Return a new cart with the line's quantity set; a non-positive quantity
    removes the line (so stepping down past 1 drops it)."""
    if quantity < 1:
        return remove_line(cart, supplier, sku)
    op = CartOp(
        op=CartOpKind.SET_QUANTITY,
        item=LineItem(sku=sku, supplier=supplier, quantity=quantity),
    )
    return cart.apply([op])


def _current_quantity(cart: Cart, supplier: str, sku: str) -> int:
    """The line's current quantity (0 if absent), matched by (supplier, sku) the
    same way ``Cart.apply`` matches lines."""
    line = next((li for li in cart.by_supplier.get(supplier, []) if li.sku == sku), None)
    return (line.quantity or 0) if line else 0


def step_line_quantity(cart: Cart, supplier: str, sku: str, delta: int) -> Cart:
    """Nudge a line's quantity by ``delta``, reading the base from the *live* cart
    rather than a value captured at render time — so a burst of −/+ clicks before
    the panel re-renders still steps consistently. Stepping below 1 drops the line
    (via :func:`set_line_quantity`)."""
    return set_line_quantity(cart, supplier, sku, _current_quantity(cart, supplier, sku) + delta)