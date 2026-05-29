"""Order agent port — the inner agent the UX boundary delegates to.

Defined outside-in: ``handle_turn`` needs *something* that takes a turn's
message plus the running cart and returns the turn's structured result. That
"something" is the LangGraph agent (deferred); this Protocol is its contract, so
the handler can be built and tested against a stub today.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from src.domain.models import Cart


class AgentResult(BaseModel):
    """One turn's structured output from the agent.

    The agent owns the cart and the decision of what to say; it returns the raw
    pieces and lets the UX boundary compose the chat reply. ``draft_cart`` is the
    full (persisted) cart after this turn — unchanged on a question turn.
    """

    draft_cart: Cart = Field(default_factory=dict)
    clarifications: list[str] = Field(default_factory=list)
    answer: str | None = None


@runtime_checkable
class OrderAgent(Protocol):
    def run(self, message: str, cart: Cart) -> AgentResult:
        """Process one turn against the running cart, returning its result."""
        ...