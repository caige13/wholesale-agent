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
from src.domain.models import Intent, LineItem, OrderStatus


class OrderState(TypedDict, total=False):
    """State for one turn of the order-desk graph."""

    raw_message: str  # original user text this turn
    clean_message: str  # redacted + unit-normalized
    pii_found: list[str]  # guardrail trace visibility (types only)

    intent: Intent | None
    line_items: list[LineItem]  # parsed + resolved + validated this turn

    draft_cart: Cart  # the running cart — PERSISTS across turns
    clarifications: Annotated[list[str], add]  # accumulates within a turn
    answer: str | None  # set on the question path; cart left untouched
    status: OrderStatus