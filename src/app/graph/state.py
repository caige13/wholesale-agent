"""OrderState — the state threaded through the LangGraph agent.

LangGraph uses a ``TypedDict`` with ``Annotated`` reducer metadata. Only
``clarifications`` accumulates within a turn (``operator.add``); everything else
is last-write-wins. That keeps this module langgraph-free — the agent dep is only
needed to *wire* the graph, not to describe its state.

``total=False`` lets the graph start from a partial dict (``raw_message`` plus
the carried-over ``draft_cart``); each node fills in its slice. ``draft_cart``
persists across turns and is mirrored by the UI.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from src.domain.cart import Cart
from src.domain.models import CartOp, Intent, OrderConfirmation, OrderStatus


class OrderState(TypedDict, total=False):
    """State for one turn of the order-desk graph."""

    raw_message: str  # original user text this turn
    clean_message: str  # redacted + unit-normalized
    pii_found: list[str]  # guardrail trace visibility (types only)
    # Recent chat turns ({role, content}) so the LLM can interpret follow-ups
    # ("yes, 16oz") against the prior question. Single-turn graph; this is the
    # lightweight alternative to a checkpointer (see README design notes).
    history: list[dict]

    intent: Intent | None
    # The per-turn operations (add/set_quantity/remove). parse_order produces them;
    # resolve/validate enrich each op's item; apply folds them into the cart.
    cart_ops: list[CartOp]
    # Add-on offers the user accepted this turn: [{"name", "quantity"}]. parse_order
    # fills it from the pending offer; add_companions turns it into ADD ops by SKU.
    accepted_companions: list[dict]

    draft_cart: Cart  # the running cart — PERSISTS across turns
    clarifications: Annotated[list[str], add]  # accumulates within a turn
    answer: str | None  # set on the question path; cart left untouched
    status: OrderStatus
    confirmation: OrderConfirmation | None  # supplier ack, set when the order drafts