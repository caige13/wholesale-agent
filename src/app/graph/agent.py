"""LangGraphOrderAgent — the OrderAgent port implemented over the compiled graph.

Adapts the LangGraph graph to the inner-agent contract the UX boundary
(``handle_turn``) depends on: takes a turn's message + the running cart, invokes
the graph, and maps the final state to an ``AgentResult``. The cart is seeded into
the initial state and returned from the final state, so a question turn (which
never writes ``draft_cart``) hands the same cart back unchanged.

``run`` uses ``invoke`` (the finished state); ``stream_run`` drives the same graph with
``graph.stream`` and yields progress + token events for the UI, then the final result.

State threading: ``run``/``stream_run`` seed ``draft_cart`` only when a cart is passed,
so when the graph was compiled with a checkpointer (production always is) the persisted
state (keyed by ``thread_id``) supplies the cart instead, and the checkpointer-owned
history carries across turns — the agent records each turn via ``record_turn``. The
``thread_id`` is always sent in the invoke config; a checkpointer-less graph (the keyless
tests) simply ignores it and runs single-turn.

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
from collections.abc import Iterator
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessageChunk

from src.domain.cart import Cart
from src.domain.redaction import redact_normalize
from src.ports.order_agent import AgentResult

if TYPE_CHECKING:
    from src.observability import TraceContext

_log = logging.getLogger(__name__)

# stream_run yields (kind, payload) events the UI renders as work happens:
#   ("progress", node_name)  — a graph node just produced output (drives a status line)
#   ("token", text)          — a chunk of the QA answer as the model writes it
#   ("result", AgentResult)  — emitted once at the end, the finished turn
StreamEvent = tuple


class LangGraphOrderAgent:
    def __init__(self, graph):
        self._graph = graph

    def run(
        self,
        message: str,
        cart: Cart | None = None,
        history: list[dict] | None = None,
        *,
        trace: TraceContext | None = None,
        thread_id: str = "default",
    ) -> AgentResult:
        config = self._config(trace, thread_id)
        try:
            final = self._graph.invoke(self._initial_state(message, cart, history), config=config)
        except Exception:
            # The turn boundary is the right place to log a failed turn once — the
            # caller (UI/eval) renders it, but the operational signal shouldn't depend
            # on tracing being on. Re-raise; don't swallow.
            _log.exception("agent turn failed")
            raise
        return self._result(final, cart)

    def stream_run(
        self,
        message: str,
        cart: Cart | None = None,
        history: list[dict] | None = None,
        *,
        trace: TraceContext | None = None,
        thread_id: str = "default",
    ) -> Iterator[StreamEvent]:
        """Drive the same graph with ``graph.stream`` and yield events as work happens.

        Yields ``("progress", node)`` as each top-level node finishes, ``("token", text)``
        for the QA answer as the model writes it (the question path only — the order
        path's reply is composed deterministically, so it has no tokens to stream), and
        finally ``("result", AgentResult)``. Same inputs/threading as :meth:`run`.
        """
        config = self._config(trace, thread_id)
        final_values: dict | None = None
        try:
            for chunk in self._graph.stream(
                self._initial_state(message, cart, history),
                config=config,
                stream_mode=["updates", "values", "messages"],
                subgraphs=True,
            ):
                namespace, mode, data = _normalize_stream_chunk(chunk)
                if mode == "updates" and not namespace:
                    for node in data or {}:  # node name(s) that produced output this step
                        yield ("progress", node)
                elif mode == "values" and not namespace:
                    final_values = data  # last parent-level snapshot = the finished state
                elif mode == "messages" and namespace:
                    # Tokens only from a subgraph (the QA assistant) — never the parent's
                    # structured-output calls, whose internal JSON must not reach the UI.
                    text = _token_text(data)
                    if text:
                        yield ("token", text)
        except Exception:
            _log.exception("agent turn failed")
            raise
        yield ("result", self._result(final_values or {}, cart))

    def record_turn(self, message: str, reply: str, *, thread_id: str = "default") -> None:
        """Append this turn to the checkpointer's history (no-op when there's no checkpointer)."""
        if getattr(self._graph, "checkpointer", None) is None:
            return
        clean = redact_normalize(message).clean_message  # keep raw PII out of persisted history
        self._graph.update_state(
            self._config(None, thread_id),
            {"history": [
                {"role": "user", "content": clean},
                {"role": "assistant", "content": reply},
            ]},
        )

    @staticmethod
    def _initial_state(message: str, cart: Cart | None, history: list[dict] | None) -> dict:
        # A new turn starts with the prior turn's per-turn OUTPUTS cleared. Under a
        # checkpointer OrderState persists across turns, so without this a question's
        # `answer` (or an escalation's `handoff`, or a placed order's `confirmation`)
        # survives into the next turn and `_compose_reply` renders it instead of this
        # turn's result. Only draft_cart and history are meant to carry over.
        state: dict = {
            "raw_message": message,
            "history": history or [],
            "answer": None,
            "handoff": None,
            "confirmation": None,
            "clarifications": [],
        }
        # Seed draft_cart ONLY when a cart is supplied; otherwise a checkpointer's
        # persisted state (keyed by thread_id) provides it, so we don't clobber it.
        if cart is not None:
            state["draft_cart"] = cart
        return state

    @staticmethod
    def _config(trace: TraceContext | None, thread_id: str) -> dict:
        # The trace labels (run_name/tags/metadata) and the checkpointer key
        # (configurable.thread_id) are orthogonal slices of one RunnableConfig.
        config: dict = dict(trace.to_runnable_config()) if trace else {}
        config["configurable"] = {**config.get("configurable", {}), "thread_id": thread_id}
        return config

    @staticmethod
    def _result(final: dict, cart: Cart | None) -> AgentResult:
        return AgentResult(
            draft_cart=final.get("draft_cart") or cart or Cart(),
            clarifications=final.get("clarifications", []),
            answer=final.get("answer"),
            confirmation=final.get("confirmation"),
            handoff=final.get("handoff"),
        )


def _normalize_stream_chunk(chunk: tuple) -> tuple:
    """Normalize a ``graph.stream`` item to ``(namespace, mode, data)``.

    With ``subgraphs=True`` and multiple stream modes, langgraph yields a 3-tuple
    ``(namespace, mode, data)``; tolerate a 2-tuple (single-mode shape) defensively.
    """
    if len(chunk) == 3:
        return chunk
    namespace, data = chunk
    return namespace, None, data


def _token_text(data) -> str:
    """Pull display text from a 'messages' stream item ``(message_chunk, metadata)``.

    Flattens Gemini's list-of-content-blocks the same way the QA finalize node does, so
    a streamed token and the final answer agree on shape.
    """
    message = data[0] if isinstance(data, tuple) else data
    # Only freshly generated tokens stream (AIMessageChunk); skip the seeded human
    # message, seeded history (full AIMessages), and tool results on this channel.
    if not isinstance(message, AIMessageChunk):
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return ""