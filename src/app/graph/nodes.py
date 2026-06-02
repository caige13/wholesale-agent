"""Deterministic graph nodes — thin wrappers over the pure domain functions.

Each node takes the current OrderState and returns a partial update LangGraph
merges in. Dependencies (the catalog retriever) are passed in; ``build_graph``
binds them into single-arg node closures. No LLM here — these are the keyless,
deterministic half of the graph. Defined in the order the graph calls them:

    redact -> (parse/intent: llm_nodes) -> resolve -> check_inventory
           -> validate -> apply -> gate -> draft | clarify
"""

from __future__ import annotations

from src.domain.cart import Cart
from src.domain.companions import companion_case_count, pending_offers
from src.domain.models import (
    CartOp,
    CartOpKind,
    CatalogItem,
    Companion,
    Flag,
    LineItem,
    OrderStatus,
)
from src.domain.policies import BLOCKING_FLAGS, CONFIDENCE_THRESHOLD
from src.domain.redaction import redact_normalize
from src.domain.resolution import resolve_skus
from src.domain.rules import validate_rules
from src.ports import CatalogRepository, SupplierGateway


def redact_node(state: dict) -> dict:
    result = redact_normalize(state["raw_message"])
    return {
        "clean_message": result.clean_message,
        "pii_found": result.pii_found,
        "status": OrderStatus.PARSING,
    }


def resolve_node(
    state: dict, catalog: CatalogRepository, item_memory: dict[str, str] | None = None
) -> dict:
    ops = []
    for op in state.get("cart_ops", []):
        candidates = catalog.find_candidates(op.item.raw_text)
        resolved = resolve_skus(op.item, candidates, item_memory)
        ops.append(op.model_copy(update={"item": resolved}))
    return {"cart_ops": ops}


def add_companions_node(state: dict, catalog: CatalogRepository) -> dict:
    """Add the add-ons the user accepted this turn — by SKU, deterministically.

    The "yes" arrives as ``accepted_companions`` (names picked from the offer). We
    preview this turn's already-resolved edits onto the cart so a simultaneous
    quantity change ("yes, and make it 6 cases") sizes the companion correctly and
    an item added now counts as in the cart, then append one fully-resolved ADD op
    per accepted offer — the user's quantity if they named one, else
    ``companion_case_count``. No fuzzy resolution: the SKU is the one we offered.
    """
    ops = list(state.get("cart_ops", []))
    accepted = {
        a["name"].strip().lower(): a.get("quantity") for a in state.get("accepted_companions", [])
    }
    if not accepted:
        return {"cart_ops": ops}
    effective = (state.get("draft_cart") or Cart()).apply([op for op in ops if op.item.sku])
    extra = []
    for parent, companion in pending_offers(effective):
        name = companion.product_name.strip().lower()
        comp_item = catalog.get(companion.sku)
        if comp_item is None or name not in accepted:
            continue
        stated = accepted[name]
        quantity = stated if stated is not None else companion_case_count(
            _parent_units(parent, catalog), comp_item.case_pack
        )
        extra.append(CartOp(op=CartOpKind.ADD, item=_line_from_catalog(comp_item, quantity)))
    return {"cart_ops": ops + extra}


def _parent_units(line: LineItem, catalog: CatalogRepository) -> int:
    """The parent line's total unit count (cases × case pack), 0 if unresolved."""
    item = catalog.get(line.sku) if line.sku else None
    return (line.quantity or 0) * (item.case_pack if item else 0)


def _line_from_catalog(item: CatalogItem, quantity: int | None) -> LineItem:
    """A fully-resolved cart line for a catalog item (used for accepted companions)."""
    return LineItem(
        raw_text=item.product_name,
        sku=item.sku,
        product_name=item.product_name,
        supplier=item.supplier,
        unit=item.unit_size,
        quantity=quantity,
        confidence=1.0,
    )


def check_inventory_node(state: dict, supplier: SupplierGateway) -> dict:
    """Enrich each resolved line with live supplier data: fill ``unit_price`` and
    raise ``OUT_OF_STOCK`` when the supplier has none. Runs before ``validate``,
    which preserves the flag/price. Removals and unresolved lines are skipped.
    """
    ops = []
    for op in state.get("cart_ops", []):
        if op.op is CartOpKind.REMOVE or not op.item.sku:
            ops.append(op)
            continue
        status = supplier.check_inventory(op.item.sku)
        updates: dict = {}
        price = supplier.get_price(op.item.sku)
        if price is not None:
            updates["unit_price"] = price
        if not status.in_stock and Flag.OUT_OF_STOCK not in op.item.flags:
            updates["flags"] = [*op.item.flags, Flag.OUT_OF_STOCK]
        if updates:
            op = op.model_copy(update={"item": op.item.model_copy(update=updates)})
        ops.append(op)
    return {"cart_ops": ops}


def validate_node(state: dict, catalog: CatalogRepository) -> dict:
    cart: Cart = state.get("draft_cart") or Cart()
    # "Already covered" = in the cart OR being added this turn (e.g. a companion
    # just accepted), so a parent never offers an add-on that's already on its way in.
    in_cart = {line.sku for line in cart.all_lines()} | {
        op.item.sku
        for op in state.get("cart_ops", [])
        if op.op is CartOpKind.ADD and op.item.sku
    }
    ops = []
    for op in state.get("cart_ops", []):
        catalog_item = catalog.get(op.item.sku) if op.item.sku else None
        # A removal isn't validated (no companions/minimum to check on the way out).
        if op.op is CartOpKind.REMOVE or catalog_item is None:
            ops.append(op)
            continue
        item = validate_rules(op.item, catalog_item)
        if Flag.NEEDS_COMPANION in item.flags:
            if op.op is CartOpKind.ADD:
                # Name the companions to offer (from the catalog), minus any already
                # in the cart — so re-adding a parent whose lids are present, or a
                # cart that already has them, raises no offer.
                item = _attach_companion_offer(item, catalog_item, catalog, in_cart)
            else:
                # The add-on nudge is an add-time suggestion; don't re-ask when only
                # the quantity of an existing line changes.
                item = item.model_copy(
                    update={"flags": _without(item.flags, Flag.NEEDS_COMPANION), "companions": []}
                )
        ops.append(op.model_copy(update={"item": item}))
    return {"cart_ops": ops}


def _attach_companion_offer(item, catalog_item, catalog: CatalogRepository, in_cart) -> LineItem:
    """Resolve catalog ``companion_skus`` to (sku, name) pairs the offer can name,
    dropping any already in the cart. If none remain, the nudge is cleared."""
    companions = [
        Companion(sku=sku, product_name=found.product_name)
        for sku in catalog_item.companion_skus
        if sku not in in_cart and (found := catalog.get(sku)) is not None
    ]
    if not companions:
        return item.model_copy(update={"flags": _without(item.flags, Flag.NEEDS_COMPANION)})
    return item.model_copy(update={"companions": companions})


def _without(flags: list[Flag], flag: Flag) -> list[Flag]:
    return [f for f in flags if f != flag]


def apply_node(state: dict) -> dict:
    cart: Cart = state.get("draft_cart") or Cart()
    # Skip unresolved lines and quantity-less adds — the gate clarifies on those
    # instead of landing a half-line ("I want salsa cups" → ask how many first).
    ops = [
        op
        for op in state.get("cart_ops", [])
        if op.item.sku and not (op.op is CartOpKind.ADD and op.item.quantity is None)
    ]
    return {"draft_cart": cart.apply(ops)}


def draft_node(state: dict, supplier: SupplierGateway) -> dict:
    """Finalize the clean (non-clarify) path. Submit to the supplier ONLY when the
    user asked to place the order (``place_order``); otherwise leave the running
    draft untouched so they can keep adding items across turns — never auto-confirm.
    """
    cart: Cart = state.get("draft_cart") or Cart()
    if not state.get("place_order") or cart.is_empty():
        return {"status": OrderStatus.DRAFTED}
    confirmation = supplier.submit_order(cart.all_lines())
    return {"status": OrderStatus.SUBMITTED, "confirmation": confirmation}


def clarify_node(state: dict) -> dict:
    items = [op.item for op in state.get("cart_ops", [])]
    return {
        "clarifications": build_clarifications(items),
        "status": OrderStatus.NEEDS_CLARIFICATION,
    }


def build_clarifications(line_items: list[LineItem]) -> list[str]:
    """One question per item the gate would stop on — low confidence or a
    blocking flag. Clean, confident items produce nothing.
    """
    questions: list[str] = []
    for line in line_items:
        label = line.product_name or line.raw_text
        if line.confidence < CONFIDENCE_THRESHOLD:
            if line.options:
                options = _join_options(line.options)
                questions.append(f'For "{line.raw_text}", did you mean {options}?')
            else:
                questions.append(
                    f'I could not confidently match "{line.raw_text}". Which product did you mean?'
                )
            continue
        blocking = [flag for flag in line.flags if flag in BLOCKING_FLAGS]
        if Flag.MISSING_QUANTITY in blocking:
            questions.append(f"How many cases of {label} would you like?")
        elif Flag.NEEDS_COMPANION in blocking:
            if line.companions:
                names = _join_options([c.product_name for c in line.companions])
                questions.append(f"{label} needs matching {names} — should I add them?")
            else:
                questions.append(f"{label} needs matching add-ons — should I add them?")
        elif Flag.OUT_OF_STOCK in blocking:
            questions.append(f"{label} is out of stock — want a substitute or to proceed?")
        elif Flag.BELOW_MINIMUM in blocking:
            questions.append(f"{label} is below its minimum order — increase the quantity?")
        elif Flag.AMBIGUOUS_SIZE in blocking:
            questions.append(f"What size of {label} did you want?")
    return questions


def _join_options(options: list[str]) -> str:
    """Render options as 'A', 'A or B', or 'A, B, or C'."""
    if len(options) == 1:
        return options[0]
    if len(options) == 2:
        return f"{options[0]} or {options[1]}"
    return f"{', '.join(options[:-1])}, or {options[-1]}"