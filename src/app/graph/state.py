"""OrderState — the state threaded through the LangGraph agent.

LangGraph uses a ``TypedDict`` with ``Annotated`` reducer metadata. Only
``history`` accumulates (``operator.add``) — it's checkpointer-owned and carries
across turns; everything else is last-write-wins and reset per turn (the agent
clears the per-turn outputs in ``_initial_state``). That keeps this module
langgraph-free — the agent dep is only needed to *wire* the graph, not to
describe its state.

``total=False`` lets the graph start from a partial dict (``raw_message`` plus
the carried-over ``draft_cart``); each node fills in its slice. ``draft_cart``
persists across turns and is mirrored by the UI.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from src.domain.cart import Cart
from src.domain.models import CartOp, Handoff, Intent, OrderConfirmation, OrderStatus


class OrderState(TypedDict, total=False):
    """State for one turn of the order-desk graph."""

    raw_message: str  # original user text this turn
    clean_message: str  # redacted + unit-normalized
    pii_found: list[str]  # guardrail trace visibility (types only)
    # Recent chat turns ({role, content}) for follow-up context. Checkpointer-owned:
    # the agent records each turn via update_state, so the UI doesn't thread it.
    history: Annotated[list[dict], add]

    intent: Intent | None
    # The per-turn operations (add/set_quantity/remove). parse_order produces them;
    # resolve/validate enrich each op's item; apply folds them into the cart.
    cart_ops: list[CartOp]
    # Add-on offers the user accepted this turn: [{"name", "quantity"}]. parse_order
    # fills it from the pending offer; add_companions turns it into ADD ops by SKU.
    accepted_companions: list[dict]
    # Set by parse_order when the user asks to finish/submit; draft submits only then,
    # otherwise the cart is left as a running draft to keep building across turns.
    place_order: bool

    draft_cart: Cart  # the running cart — PERSISTS across turns
    clarifications: list[str]  # per-turn; reset each turn (see agent._initial_state)
    answer: str | None  # set on the question path; cart left untouched
    status: OrderStatus
    confirmation: OrderConfirmation | None  # supplier ack, set when the order drafts
    handoff: Handoff | None  # human-handoff ticket, set on the escalation branch