"""handle_turn — the UX boundary the Gradio callback delegates to.

The UI owns no logic (spec §11): it hands the user's message and the current
cart here, and renders the returned ``TurnResult`` (chat reply + cart panel).
This layer's job is to compose the human-facing reply from the agent's
structured result and to guarantee the cart is always returned — so a question
turn never blanks the cart panel.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.cart import Cart
from src.domain.models import LineItem
from src.ports.order_agent import AgentResult, OrderAgent


class TurnResult(BaseModel):
    """What the front-end renders for one turn: a chat reply and the cart to show."""

    reply: str
    cart: Cart = Field(default_factory=Cart)


def handle_turn(
    message: str, cart: Cart, agent: OrderAgent, history: list[dict] | None = None
) -> TurnResult:
    """Run one turn through the agent and shape it for the UI.

    ``history`` (recent {role, content} turns) is passed through so the agent can
    resolve follow-ups against the prior exchange.
    """
    # Pass history only when present, so an agent whose run() predates the
    # history parameter keeps working — the UI adopts history at its own pace.
    result = agent.run(message, cart, history) if history else agent.run(message, cart)
    return TurnResult(reply=_compose_reply(result), cart=result.draft_cart)


def _compose_reply(result: AgentResult) -> str:
    """Pick the one thing to say this turn: an answer, a clarifying question, or
    a draft summary. Order matters — a pending question or answer trumps the cart.
    """
    answer = (result.answer or "").strip()
    if answer:
        return answer
    if result.clarifications:
        return "\n".join(result.clarifications)
    summary = _summarize_cart(result.draft_cart)
    if result.confirmation and not result.draft_cart.is_empty():
        summary += f"\nOrder confirmed: {result.confirmation.order_id}"
    return summary


def _summarize_cart(cart: Cart) -> str:
    if cart.is_empty():
        return "Your cart is empty — tell me what you'd like to order."
    rendered = "\n".join(_render_line(item) for item in cart.all_lines())
    return "Here's your draft order:\n" + rendered


def _render_line(item: LineItem) -> str:
    """Render one cart line, degrading gracefully for a not-yet-resolved line so
    a missing name/quantity never surfaces as the literal "None"."""
    label = item.product_name or item.sku or "item"
    if item.quantity is None:
        return f"- {label}"
    return f"- {item.quantity} × {label}"