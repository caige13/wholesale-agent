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
from src.domain.companions import companion_case_count
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
from src.ports import CatalogRepository, EscalationGateway, SupplierGateway


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
    """Apply the add-ons the user accepted this turn — by SKU, deterministically.

    The "yes" arrives as ``accepted_companions`` (names picked from the offer). We
    preview this turn's already-resolved edits so coverage reflects them, then for
    each accepted companion emit a SET_QUANTITY to the total that covers *all* the
    deli lines it pairs with (so adding a second deli size tops the lid up rather
    than under-covering) — or the user's amount if they named one. SET_QUANTITY
    appends when the companion is new and replaces when it's already in the cart.
    No fuzzy resolution: the SKU is the one we offered.
    """
    ops = list(state.get("cart_ops", []))
    accepted = {
        a["name"].strip().lower(): a.get("quantity") for a in state.get("accepted_companions", [])
    }
    if not accepted:
        return {"cart_ops": ops}
    effective = (state.get("draft_cart") or Cart()).apply([op for op in ops if op.item.sku])
    extra = []
    for companion, needed in _undercovered_companions(effective, catalog):
        if companion.product_name.strip().lower() not in accepted:
            continue
        comp_item = catalog.get(companion.sku)
        if comp_item is None:
            continue
        stated = accepted[companion.product_name.strip().lower()]
        quantity = stated if stated is not None else needed
        extra.append(
            CartOp(op=CartOpKind.SET_QUANTITY, item=_line_from_catalog(comp_item, quantity))
        )
    return {"cart_ops": ops + extra}


def _companion_coverage(
    cart: Cart, catalog: CatalogRepository
) -> dict[str, tuple[Companion, int, int]]:
    """Per companion SKU referenced by the cart: (Companion, needed_cases, current_cases).

    ``needed`` covers the summed units of *every* line that pairs with it (a lid fits
    all deli sizes); ``current`` is that companion's own quantity in the cart.
    """
    current: dict[str, int] = {}
    units: dict[str, int] = {}
    for line in cart.all_lines():
        if line.sku:
            current[line.sku] = current.get(line.sku, 0) + (line.quantity or 0)
        item = catalog.get(line.sku) if line.sku else None
        if item is None:
            continue
        for companion_sku in item.companion_skus:
            units[companion_sku] = (
                units.get(companion_sku, 0) + (line.quantity or 0) * item.case_pack
            )
    coverage: dict[str, tuple[Companion, int, int]] = {}
    for companion_sku, total_units in units.items():
        comp = catalog.get(companion_sku)
        if comp is None:
            continue
        needed = companion_case_count(total_units, comp.case_pack)
        name = Companion(sku=comp.sku, product_name=comp.product_name)
        coverage[companion_sku] = (name, needed, current.get(companion_sku, 0))
    return coverage


def _undercovered_companions(cart: Cart, catalog: CatalogRepository) -> list[tuple[Companion, int]]:
    """(Companion, needed_cases) for companions the cart doesn't yet cover."""
    return [
        (companion, needed)
        for companion, needed, current in _companion_coverage(cart, catalog).values()
        if needed > current
    ]


def pending_companions(cart: Cart, catalog: CatalogRepository) -> list[Companion]:
    """Still-open add-on offers for the parse prompt — companions the cart under-covers."""
    return [companion for companion, _ in _undercovered_companions(cart, catalog)]


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
    base: Cart = state.get("draft_cart") or Cart()
    # Coverage is a cart-level property, so preview this turn's resolved edits, then
    # a companion is worth offering only if it still UNDER-covers all the lines it
    # pairs with — re-adding a parent whose lids are already adequate raises nothing,
    # while adding a second deli size that outgrows the lids re-offers a top-up.
    effective = base.apply([op for op in state.get("cart_ops", []) if op.item.sku])
    undercovered = {companion.sku for companion, _ in _undercovered_companions(effective, catalog)}
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
                item = _attach_companion_offer(item, catalog_item, catalog, undercovered)
            else:
                # The add-on nudge is an add-time suggestion; don't re-ask when only
                # the quantity of an existing line changes.
                item = item.model_copy(
                    update={"flags": _without(item.flags, Flag.NEEDS_COMPANION), "companions": []}
                )
        ops.append(op.model_copy(update={"item": item}))
    return {"cart_ops": ops}


def _attach_companion_offer(item, catalog_item, catalog, undercovered: set[str]) -> LineItem:
    """Attach the catalog companions that are still under-covered (named for the
    offer). If the cart already covers them all, clear the nudge."""
    companions = [
        Companion(sku=sku, product_name=found.product_name)
        for sku in catalog_item.companion_skus
        if sku in undercovered and (found := catalog.get(sku)) is not None
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


def escalate_node(state: dict, escalation: EscalationGateway) -> dict:
    """Hand the turn off to a human — the terminal escalation branch.

    Reached when intent routing classifies the turn as ESCALATE (the user asked for a
    person, or for something the order desk can't do). Opens a handoff ticket via the
    gateway and stops: the cart is left untouched (no order mutation on the way out),
    and the UI renders the ticket so the customer knows a specialist will follow up.
    """
    handoff = escalation.create_handoff(
        reason="customer requested a human or an unsupported action",
        summary=state.get("clean_message", ""),
    )
    return {"handoff": handoff, "status": OrderStatus.ESCALATED}


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