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
from src.domain.models import CartOp, CartOpKind, Flag, LineItem, OrderStatus
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
    resolved = [
        resolve_skus(line, catalog.find_candidates(line.raw_text), item_memory)
        for line in state.get("line_items", [])
    ]
    return {"line_items": resolved}


def validate_node(state: dict, catalog: CatalogRepository) -> dict:
    validated = []
    for line in state.get("line_items", []):
        catalog_item = catalog.get(line.sku) if line.sku else None
        validated.append(validate_rules(line, catalog_item) if catalog_item else line)
    return {"line_items": validated}


def apply_node(state: dict) -> dict:
    cart: Cart = state.get("draft_cart") or Cart()
    ops = [
        CartOp(op=CartOpKind.ADD, item=line)
        for line in state.get("line_items", [])
        if line.sku
    ]
    return {"draft_cart": cart.apply(ops)}


def draft_node(state: dict) -> dict:
    return {"status": OrderStatus.DRAFTED}


def clarify_node(state: dict) -> dict:
    return {
        "clarifications": build_clarifications(state.get("line_items", [])),
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