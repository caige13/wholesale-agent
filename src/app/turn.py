"""handle_turn — the UX boundary the Gradio callback delegates to.

The UI owns no logic (spec §11): it hands the user's message and the current
cart here, and renders the returned ``TurnResult`` (chat reply + cart panel).
This layer's job is to compose the human-facing reply from the agent's
structured result and to guarantee the cart is always returned — so a question
turn never blanks the cart panel.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import Cart
from src.ports.order_agent import AgentResult, OrderAgent


class TurnResult(BaseModel):
    """What the front-end renders for one turn: a chat reply and the cart to show."""

    reply: str
    cart: Cart = Field(default_factory=dict)


def handle_turn(message: str, cart: Cart, agent: OrderAgent) -> TurnResult:
    """Run one turn through the agent and shape it for the UI."""
    result = agent.run(message, cart)
    return TurnResult(reply=_compose_reply(result), cart=result.draft_cart)


def _compose_reply(result: AgentResult) -> str:
    """Pick the one thing to say this turn: an answer, a clarifying question, or
    a draft summary. Order matters — a pending question or answer trumps the cart.
    """
    if result.answer:
        return result.answer
    if result.clarifications:
        return "\n".join(result.clarifications)
    return _summarize_cart(result.draft_cart)


def _summarize_cart(cart: Cart) -> str:
    lines = [item for items in cart.values() for item in items]
    if not lines:
        return "Your cart is empty — tell me what you'd like to order."
    rendered = "\n".join(f"- {item.quantity} × {item.product_name}" for item in lines)
    return "Here's your draft order:\n" + rendered