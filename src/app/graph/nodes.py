"""Deterministic graph nodes — thin wrappers over the pure domain functions.

Each node takes the current OrderState and returns a partial update LangGraph
merges in. Dependencies (the catalog retriever) are passed in; ``build_graph``
binds them into single-arg node closures. No LLM here — these are the keyless,
deterministic half of the graph. Defined in the order the graph calls them:

    redact -> (parse/intent: llm_nodes) -> resolve -> validate -> apply
           -> gate -> draft | clarify
"""

from __future__ import annotations

from src.domain.cart import Cart
from src.domain.models import CartOpKind, Flag, LineItem, OrderStatus
from src.domain.policies import BLOCKING_FLAGS, CONFIDENCE_THRESHOLD
from src.domain.redaction import redact_normalize
from src.domain.resolution import resolve_skus
from src.domain.rules import validate_rules
from src.ports import CatalogRepository


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


def validate_node(state: dict, catalog: CatalogRepository) -> dict:
    ops = []
    for op in state.get("cart_ops", []):
        catalog_item = catalog.get(op.item.sku) if op.item.sku else None
        # A removal isn't validated (no lids/minimum to check on the way out).
        if op.op is CartOpKind.REMOVE or catalog_item is None:
            ops.append(op)
            continue
        item = validate_rules(op.item, catalog_item)
        # The lids nudge is an add-time suggestion; don't re-ask when only the
        # quantity of an existing line changes.
        if op.op is not CartOpKind.ADD and Flag.NEEDS_LIDS in item.flags:
            kept = [f for f in item.flags if f != Flag.NEEDS_LIDS]
            item = item.model_copy(update={"flags": kept})
        ops.append(op.model_copy(update={"item": item}))
    return {"cart_ops": ops}


def apply_node(state: dict) -> dict:
    cart: Cart = state.get("draft_cart") or Cart()
    ops = [op for op in state.get("cart_ops", []) if op.item.sku]  # skip unresolved
    return {"draft_cart": cart.apply(ops)}


def draft_node(state: dict) -> dict:
    return {"status": OrderStatus.DRAFTED}


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
        if Flag.NEEDS_LIDS in blocking:
            questions.append(f"{label} needs matching lids — should I add them?")
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