"""LangGraphOrderAgent — the OrderAgent port implemented over the compiled graph.

Adapts the LangGraph graph to the inner-agent contract the UX boundary
(``handle_turn``) depends on: takes a turn's message + the running cart, invokes
the graph, and maps the final state to an ``AgentResult``. The cart is seeded into
the initial state and returned from the final state, so a question turn (which
never writes ``draft_cart``) hands the same cart back unchanged.

Uses ``invoke`` (we need the finished state). Token/step streaming to the UI is
a Gradio-layer concern — that path will use ``graph.stream`` instead.

Observability lives only here, at the boundary — never in the (pure) nodes. When a
``TraceContext`` is supplied we label the run (name + tags + metadata) via the invoke
config; that's set at run creation, so it's conflict-free and free. The turn's outcome
(intent / status / clarifications) needs no extra call — it's already captured as the
run's outputs (keys in the final state). Keyless callers pass no trace.

Tracing and logging are complementary, not substitutes: the trace (when enabled) has
the full failing run, but it's optional and remote, so a turn that blows up — LLM
error, rate limit, malformed structured output — is also logged here as an always-on
operational signal, then re-raised for the caller to render.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.domain.cart import Cart
from src.ports.order_agent import AgentResult

if TYPE_CHECKING:
    from src.observability import TraceContext

_log = logging.getLogger(__name__)


class LangGraphOrderAgent:
    def __init__(self, graph):
        self._graph = graph

    def run(
        self,
        message: str,
        cart: Cart,
        history: list[dict] | None = None,
        *,
        trace: TraceContext | None = None,
    ) -> AgentResult:
        state = {"raw_message": message, "draft_cart": cart, "history": history or []}
        config = trace.to_runnable_config() if trace else None
        try:
            final = self._graph.invoke(state, config=config)
        except Exception:
            # The turn boundary is the right place to log a failed turn once — the
            # caller (UI/eval) renders it, but the operational signal shouldn't depend
            # on tracing being on. Re-raise; don't swallow.
            _log.exception("agent turn failed")
            raise
        return AgentResult(
            draft_cart=final.get("draft_cart") or cart,
            clarifications=final.get("clarifications", []),
            answer=final.get("answer"),
            confirmation=final.get("confirmation"),
        )