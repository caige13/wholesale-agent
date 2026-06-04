"""handle_turn — the UX boundary the Gradio callback delegates to.

The UI owns no logic (spec §11): it hands the user's message and the current
cart here, and renders the returned ``TurnResult`` (chat reply + cart panel).
This layer's job is to compose the human-facing reply from the agent's
structured result and to guarantee the cart is always returned — so a question
turn never blanks the cart panel.

``handle_turn`` is the blocking one-shot; ``stream_turn`` is its streaming sibling —
it yields ``StreamFrame``s (a progress line, then the answer as it's written, then the
final composed reply) so the chat shows real progress instead of a post-hoc typewriter.
Both compose the reply the same way, so a streamed turn and a blocking turn agree.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.domain.cart import Cart
from src.domain.models import LineItem
from src.ports.order_agent import AgentResult, OrderAgent

if TYPE_CHECKING:
    from src.observability import TraceContext


class TurnResult(BaseModel):
    """What the front-end renders for one turn: a chat reply and the cart to show."""

    reply: str
    cart: Cart = Field(default_factory=Cart)


class StreamFrame(BaseModel):
    """One frame of a streamed turn: the reply-so-far, the cart, and which phase it's in.

    Intermediate frames carry a progress line or the partial answer and the *prior*
    cart (the panel shouldn't flicker mid-turn); the final frame (``done=True``) carries
    the fully composed reply and the new cart for the UI to commit. ``phase`` lets the
    front-end style the frame — an ephemeral progress line vs. the answer vs. the reply.
    """

    reply: str
    cart: Cart = Field(default_factory=Cart)
    done: bool = False
    phase: str = "final"  # "progress" | "answer" | "final"


# Friendly status lines per graph node, shown while that node runs (streamed turns).
_PROGRESS_LABELS = {
    "redact": "Reading your message…",
    "intent": "Reading your message…",
    "parse": "Building your order…",
    "resolve": "Matching products…",
    "add_companions": "Building your order…",
    "check_inventory": "Checking stock and pricing…",
    "validate": "Checking the order…",
    "apply": "Updating your cart…",
    "rag_qa": "Looking that up…",
    "clarify": "Putting together a question…",
    "draft": "Finalizing…",
    "escalate": "Connecting you with a specialist…",
}


def handle_turn(
    message: str,
    cart: Cart,
    agent: OrderAgent,
    history: list[dict] | None = None,
    *,
    trace: TraceContext | None = None,
    thread_id: str = "default",
) -> TurnResult:
    """Run one turn through the agent and shape it for the UI.

    Two independent pass-throughs: ``history`` (recent {role, content} turns) is a
    *behavioral* input — it feeds the prompt so the agent can resolve follow-ups;
    ``trace`` is *observability* metadata the agent attaches to the LangSmith run and
    that never changes the result. They're orthogonal, so neither branches on the
    other — both are forwarded as-is. ``thread_id`` keys the checkpointer's per-turn state.
    """
    result = agent.run(message, cart, history, trace=trace, thread_id=thread_id)
    reply = _compose_reply(result)
    agent.record_turn(message, reply, thread_id=thread_id)
    return TurnResult(reply=reply, cart=result.draft_cart)


def stream_turn(
    message: str,
    cart: Cart,
    agent,
    history: list[dict] | None = None,
    *,
    trace: TraceContext | None = None,
    thread_id: str = "default",
) -> Iterator[StreamFrame]:
    """Stream one turn: progress, then the answer as it's written, then the final reply.

    Translates the agent's ``stream_run`` events into UI frames. The cart panel is only
    meant to update on the final frame, so intermediate frames carry the prior ``cart``
    and ``done=False``; the terminal frame carries the composed reply and the new cart.
    Requires a streaming-capable agent (``stream_run``); the blocking path is ``handle_turn``.
    """
    answer: list[str] = []
    reply = ""
    for kind, payload in agent.stream_run(message, cart, history, trace=trace, thread_id=thread_id):
        if kind == "progress":
            label = _PROGRESS_LABELS.get(payload)
            if label and not answer:  # once the answer is streaming, trailing progress is stale
                yield StreamFrame(reply=label, cart=cart, phase="progress")
        elif kind == "token":
            answer.append(payload)
            yield StreamFrame(reply="".join(answer), cart=cart, phase="answer")
        elif kind == "result":
            reply = _compose_reply(payload)
            yield StreamFrame(reply=reply, cart=payload.draft_cart, done=True, phase="final")
    agent.record_turn(message, reply, thread_id=thread_id)


def _compose_reply(result: AgentResult) -> str:
    """Pick the one thing to say this turn: a human handoff, an answer, a clarifying
    question, or a draft summary. Order matters — a handoff or answer trumps the cart.
    """
    if result.handoff:
        return _render_handoff(result.handoff)
    answer = (result.answer or "").strip()
    if answer:
        return answer
    if result.clarifications:
        return "\n".join(result.clarifications)
    summary = _summarize_cart(result.draft_cart)
    if result.draft_cart.is_empty():
        return summary
    if result.confirmation:
        return summary + f"\nOrder confirmed: {result.confirmation.order_id}"
    # A running draft, not yet placed — invite the next item or an explicit checkout.
    return summary + "\nAnything else, or should I place the order?"


def _render_handoff(handoff) -> str:
    """The chat reply for an escalation turn — the handoff ticket, callback, and ETA."""
    return (
        "I've connected you with a human specialist about that — your reference is ticket "
        f"{handoff.ticket_id}. A specialist will follow up within about {handoff.eta_minutes} "
        f"minutes; you can also reach our team directly at {handoff.callback_number}."
    )


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