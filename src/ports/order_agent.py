"""Order agent port — the inner agent the UX boundary delegates to.

Defined outside-in: ``handle_turn`` needs *something* that takes a turn's
message plus the running cart and returns the turn's structured result. That
"something" is the LangGraph agent, implemented in ``src/app/graph/agent.py``
(``LangGraphOrderAgent``); this Protocol is its contract, so the handler can be
built and tested against a stub today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from src.domain.cart import Cart
from src.domain.models import Handoff, OrderConfirmation

if TYPE_CHECKING:
    from src.observability import TraceContext


class AgentResult(BaseModel):
    """One turn's structured output from the agent.

    The agent owns the cart and the decision of what to say; it returns the raw
    pieces and lets the UX boundary compose the chat reply. ``draft_cart`` is the
    full (persisted) cart after this turn — unchanged on a question turn.
    ``confirmation`` is set only when the order drafts cleanly (the supplier ack);
    ``handoff`` only on the escalation branch (the human-handoff ticket).
    """

    draft_cart: Cart = Field(default_factory=Cart)
    clarifications: list[str] = Field(default_factory=list)
    answer: str | None = None
    confirmation: OrderConfirmation | None = None
    handoff: Handoff | None = None


@runtime_checkable
class OrderAgent(Protocol):
    def run(
        self,
        message: str,
        cart: Cart | None = None,
        history: list[dict] | None = None,
        *,
        selected_suppliers: list[str] | None = None,
        trace: TraceContext | None = None,
        thread_id: str = "default",
    ) -> AgentResult:
        """Process one turn against the running cart, returning its result.

        ``cart`` seeds the running cart; it may be omitted when a checkpointer holds
        it (the persisted state under ``thread_id`` supplies it instead). ``history``
        is the recent conversation ({role, content}); it lets the agent interpret
        follow-ups in context (e.g. answering a clarification). ``selected_suppliers``
        scopes SKU resolution to the customer's chosen suppliers (multi-tenant);
        ``None`` searches all. ``trace`` is an optional observability context the
        boundary attaches to the LangSmith run; concrete agents may ignore it.
        ``thread_id`` keys the checkpointer's per-conversation state.
        """
        ...

    def record_turn(self, message: str, reply: str, *, thread_id: str = "default") -> None:
        """Persist one finished turn (user message + composed reply) into the agent's own
        history, so the next turn has context without the UI threading it. No-op without
        persistence.
        """
        ...