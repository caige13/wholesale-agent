"""LangGraphOrderAgent — the OrderAgent port implemented over the compiled graph.

Adapts the LangGraph graph to the inner-agent contract the UX boundary
(``handle_turn``) depends on: takes a turn's message + the running cart, invokes
the graph, and maps the final state to an ``AgentResult``. The cart is seeded into
the initial state and returned from the final state, so a question turn (which
never writes ``draft_cart``) hands the same cart back unchanged.

Uses ``invoke`` (we need the finished state). Token/step streaming to the UI is
a Gradio-layer concern — that path will use ``graph.stream`` instead.
"""

from __future__ import annotations

from src.domain.cart import Cart
from src.ports.order_agent import AgentResult


class LangGraphOrderAgent:
    def __init__(self, graph):
        self._graph = graph

    def run(self, message: str, cart: Cart) -> AgentResult:
        final = self._graph.invoke({"raw_message": message, "draft_cart": cart})
        return AgentResult(
            draft_cart=final.get("draft_cart") or cart,
            clarifications=final.get("clarifications", []),
            answer=final.get("answer"),
        )