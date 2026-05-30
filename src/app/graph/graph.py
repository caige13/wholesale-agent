r"""build_graph — wire the order-desk LangGraph StateGraph.

Control flow (solid v1 path):

    START -> redact -> intent --(question)--> rag_qa -----------------+
                              \--(order/reorder)--> parse -> resolve   |
                                  -> validate -> apply -> [gate]       |
                                       gate --(clarify)--> clarify ----+--> END
                                       gate --(draft)----> draft ------+

Dependencies (the chat model, the catalog retriever) are injected and bound into
single-arg node closures here — the composition root (``bootstrap``) supplies the
real ones; tests supply fakes.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.app.graph.gates import gate
from src.app.graph.llm_nodes import intent_node, parse_node, rag_qa_node
from src.app.graph.nodes import (
    apply_node,
    clarify_node,
    draft_node,
    redact_node,
    resolve_node,
    validate_node,
)
from src.app.graph.state import OrderState
from src.domain.models import Intent
from src.ports import CatalogRepository


def build_graph(model, catalog: CatalogRepository, item_memory: dict[str, str] | None = None):
    """Compile the order-desk graph with the given model + catalog injected."""
    builder = StateGraph(OrderState)

    builder.add_node("redact", redact_node)
    builder.add_node("intent", lambda state: intent_node(state, model))
    builder.add_node("parse", lambda state: parse_node(state, model))
    builder.add_node("resolve", lambda state: resolve_node(state, catalog, item_memory))
    builder.add_node("validate", lambda state: validate_node(state, catalog))
    builder.add_node("apply", apply_node)
    builder.add_node("rag_qa", lambda state: rag_qa_node(state, model, catalog))
    builder.add_node("clarify", clarify_node)
    builder.add_node("draft", draft_node)

    builder.add_edge(START, "redact")
    builder.add_edge("redact", "intent")
    builder.add_conditional_edges(
        "intent", _route_by_intent, {"rag_qa": "rag_qa", "parse": "parse"}
    )
    builder.add_edge("parse", "resolve")
    builder.add_edge("resolve", "validate")
    builder.add_edge("validate", "apply")
    builder.add_conditional_edges(
        "apply", _gate_decision, {"clarify": "clarify", "draft": "draft"}
    )
    builder.add_edge("rag_qa", END)
    builder.add_edge("clarify", END)
    builder.add_edge("draft", END)

    return builder.compile()


def _route_by_intent(state: OrderState) -> str:
    return "rag_qa" if state.get("intent") is Intent.QUESTION else "parse"


def _gate_decision(state: OrderState) -> str:
    return gate([op.item for op in state.get("cart_ops", [])])