"""The gradio-free UI seam the Gradio callback delegates to.

``run_turn`` is the callback's core — it runs one turn and folds it into the chat
history; ``_initial_history`` / ``_new_thread`` seed a fresh session. This module
imports no gradio (gradio lives only in the ``agent`` extra), so the seam stays
unit-testable keyless under ``make setup`` — the deterministic nodes run for free
and a scripted fake covers the LLM nodes.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from src.app.turn import handle_turn
from src.domain.cart import Cart
from src.ports.order_agent import OrderAgent

if TYPE_CHECKING:
    from src.observability import TraceContext

_EXAMPLES = [
    "3 cases of 16oz deli containers and some salsa cups",
    "how many per case?",
    "make the deli containers 5 cases",
    "drop the salsa cups",
    "I need some deli containers",
]

# A friendly opening line so the chat isn't a blank slate on load. It's seeded as
# the first assistant bubble and restored by "New ticket". The seam tests pass
# their own history, so this only shapes the live shell; once the user replies it
# simply rides along as the conversation's opening turn (harmless context).
_GREETING = (
    "Welcome to the Order Desk. Tell me what you need and I'll draft the order — "
    "say something like “3 cases of 16oz deli containers and some salsa cups.” "
    "I can also answer questions about pack sizes, pricing, and availability."
)


def _initial_history() -> list[dict]:
    """The chat's seeded state: a single assistant greeting bubble."""
    return [{"role": "assistant", "content": _GREETING}]


def _new_thread() -> str:
    """A fresh checkpointer thread id — one per browser session and per New ticket, so
    the agent's persisted state is isolated per conversation rather than shared."""
    return uuid.uuid4().hex


def run_turn(
    agent: OrderAgent,
    message: str,
    history: list[dict],
    cart: Cart,
    *,
    trace: TraceContext | None = None,
    thread_id: str = "default",
) -> tuple[list[dict], Cart]:
    """Run one turn and fold it into the chat history.

    Returns the message list (messages format) with this turn appended and the
    cart the agent handed back — always ``result.cart``, so a question turn keeps
    the panel intact (spec §11). The prior ``history`` is forwarded so the agent
    can resolve follow-ups against the conversation so far; ``trace`` (if any) labels
    the LangSmith run; ``thread_id`` keys the checkpointer's per-session state.
    """
    result = handle_turn(message, cart, agent, history, trace=trace, thread_id=thread_id)
    new_history = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.reply},
    ]
    return new_history, result.cart


def _ui_trace(history_len: int) -> TraceContext:
    """A ``ui``-surface TraceContext for one turn (import kept lazy/keyless-safe)."""
    from src.observability import TraceContext

    return TraceContext(surface="ui", metadata={"history_len": history_len})