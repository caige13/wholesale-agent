"""Cart → packing-slip HTML — the read-only slip markup (also the CLI/print view).

These pure string helpers turn a :class:`~src.domain.cart.Cart` into the warehouse
packing-slip markup. ``render_cart`` is the standalone view (and part of the
unit-tested UI seam); the interactive panel in :mod:`.app` reuses the same snippet
helpers (``_slip_header_html``, ``_line_text_html``, …) but swaps the static
quantity for a stepper, so the two views stay in lockstep.
"""

from __future__ import annotations

import html

from src.domain.cart import Cart
from src.domain.models import Flag, LineItem
from src.domain.policies import BLOCKING_FLAGS


def render_cart(cart: Cart) -> str:
    """Render the cart as a read-only packing slip (also the CLI/print view).

    The interactive panel (:func:`build_app`) reuses the same snippet helpers but
    swaps the static quantity for a stepper, so the two views stay in lockstep.
    """
    if cart.is_empty():
        return _EMPTY_SLIP

    groups = "".join(
        _render_group(supplier, items) for supplier, items in cart.by_supplier.items()
    )
    return (
        '<div class="slip">'
        + _slip_header_html(cart)
        + f'<div class="slip__body">{groups}</div>'
        + _footer_html(cart)
        + "</div>"
    )


def _slip_header_html(cart: Cart) -> str:
    lines, suppliers = cart.all_lines(), cart.by_supplier
    meta = f"{len(lines)} line{_s(lines)} · {len(suppliers)} supplier{_s(suppliers)}"
    return (
        '<div class="slip__head">'
        '<span class="slip__title">Packing Slip</span>'
        f'<span class="slip__meta">{meta}</span>'
        "</div>"
        '<div class="slip__perf"></div>'
    )


def _footer_html(cart: Cart) -> str:
    return (
        '<div class="slip__perf"></div>'
        '<div class="slip__foot">'
        '<span class="slip__foot-label">Indicative subtotal</span>'
        f'<span class="slip__foot-val">{_subtotal(cart.all_lines())}</span>'
        "</div>"
        '<div class="slip__note">Final pricing confirmed at submit.</div>'
    )


def _group_header_html(supplier: str | None) -> str:
    return f'<div class="grp__name">{html.escape(supplier or "unassigned")}</div>'


def _render_group(supplier: str | None, items: list[LineItem]) -> str:
    rows = "".join(_render_line(item) for item in items)
    return f'<div class="grp">{_group_header_html(supplier)}{rows}</div>'


def _render_line(item: LineItem) -> str:
    qty = str(item.quantity) if item.quantity is not None else "—"
    return (
        '<div class="ln">'
        f'<div class="ln__qty">{html.escape(qty)}<span class="ln__x">×</span></div>'
        f"{_line_text_html(item)}"
        "</div>"
    )


def _line_text_html(item: LineItem) -> str:
    """The name / sub-line / flags / price of a line — everything but the quantity,
    so the interactive panel can render this beside its stepper."""
    label = html.escape(item.product_name or item.sku or "item")

    sub_bits = [b for b in (item.sku, item.unit) if b]
    sub = html.escape(" · ".join(sub_bits))
    sub_html = f'<div class="ln__sub">{sub}</div>' if sub else ""

    flags_html = ""
    if item.flags:
        chips = "".join(_render_chip(flag) for flag in item.flags)
        flags_html = f'<div class="ln__flags">{chips}</div>'

    price_html = ""
    if item.unit_price is not None:
        price_html = f'<div class="ln__price">${item.unit_price:.2f} ea</div>'

    return (
        '<div class="ln__main">'
        f'<div class="ln__name">{label}</div>'
        f"{sub_html}{flags_html}"
        "</div>"
        f"{price_html}"
    )


def _render_chip(flag: Flag) -> str:
    # Blocking flags read as warnings on the slip; everything else (only
    # ROUNDED_TO_CASE_PACK today) is informational. Classification is owned by
    # src.domain.policies so the UI can't drift from the gate.
    kind = "warn" if flag in BLOCKING_FLAGS else "info"
    text = html.escape(flag.value.replace("_", " "))
    return f'<span class="chip chip--{kind}">{text}</span>'


def _subtotal(lines: list[LineItem]) -> str:
    priced = [
        line.quantity * line.unit_price
        for line in lines
        if line.quantity is not None and line.unit_price is not None
    ]
    return f"${sum(priced):.2f}" if priced else "—"


def _s(seq) -> str:
    return "" if len(seq) == 1 else "s"


# The empty-state contents, sans card. The interactive panel renders this directly
# (its column is already the card); render_cart wraps it in the .slip card below.
_EMPTY_SLIP_INNER = (
    '<div class="stamp">Awaiting Order</div>'
    '<p class="slip__hint">Tell the desk what you need —<br>'
    "“3 cases of 16oz deli containers and some salsa cups.”</p>"
)

_EMPTY_SLIP = f'<div class="slip slip--empty">{_EMPTY_SLIP_INNER}</div>'