r"""build_graph — wire the order-desk LangGraph StateGraph.

Control flow (solid v1 path):

    START -> redact -> intent --(question)--> rag_qa --------------------------+
                              |--(escalate)--> escalate --------------------------+
                              \--(order/reorder)--> parse -> resolve            |
                                  -> add_companions -> check_inventory          |
                                  -> validate -> apply -> [gate]                |
                                       gate --(clarify)--> clarify -------------+--> END
                                       gate --(draft)----> draft ---------------+

    ``escalate`` is the human-handoff branch: when intent routing decides the user wants
    a person (or something the desk can't do), it opens a handoff ticket and ends — the
    cart is never touched. ``rag_qa`` is not a single node but a compiled tool-calling
    **subgraph** (see
    ``subgraphs/qa_agent``): the model is bound to read-only catalog/supplier tools and
    loops assistant<->tools until it answers. The order path stays deterministic — only
    this read-only question branch lets the model drive tool calls.

    add_companions turns an accepted add-on offer into ADD ops (by SKU) right after
    resolve, so they ride through pricing/validation/apply like any other line.

Dependencies (the chat model, the catalog retriever) are injected and bound into
single-arg node closures here — the composition root (``bootstrap``) supplies the
real ones; tests supply fakes.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.app.graph.gates import gate
from src.app.graph.llm_nodes import intent_node, parse_node
from src.app.graph.nodes import (
    add_companions_node,
    apply_node,
    check_inventory_node,
    clarify_node,
    draft_node,
    escalate_node,
    redact_node,
    resolve_node,
    validate_node,
)
from src.app.graph.state import OrderState
from src.app.graph.subgraphs.qa_agent import build_qa_agent
from src.app.graph.subgraphs.tools import build_escalation_tool, build_order_desk_tools
from src.domain.models import Intent
from src.ports import CatalogRepository, EscalationGateway, SupplierGateway


def build_graph(
    model,
    catalog: CatalogRepository,
    supplier: SupplierGateway,
    item_memory: dict[str, str] | None = None,
    escalation: EscalationGateway | None = None,
    checkpointer=None,
):
    """Compile the order-desk graph with the given dependencies injected.

    ``escalation`` (the human-handoff gateway) enables the ESCALATE branch and adds the
    model-callable ``escalate_to_human`` tool to the question path; tests that don't
    exercise escalation may omit it. ``checkpointer`` (e.g. a ``MemorySaver``) persists
    ``OrderState`` per ``thread_id`` across turns — omitted, the graph is single-turn and
    the caller threads the cart/history explicitly (the keyless suite's default).
    """
    builder = StateGraph(OrderState)

    builder.add_node("redact", redact_node)
    builder.add_node("intent", lambda state: intent_node(state, model))
    builder.add_node("parse", lambda state: parse_node(state, model, catalog))
    builder.add_node("resolve", lambda state: resolve_node(state, catalog, item_memory))
    builder.add_node("add_companions", lambda state: add_companions_node(state, catalog))
    builder.add_node("check_inventory", lambda state: check_inventory_node(state, supplier))
    builder.add_node("validate", lambda state: validate_node(state, catalog))
    builder.add_node("apply", apply_node)
    # The QUESTION path is a real tool-calling agent compiled as its own subgraph and
    # embedded here as a single node — read-only catalog/supplier tools, model-driven.
    # When an escalation gateway is wired, the model can also call escalate_to_human.
    qa_tools = build_order_desk_tools(catalog, supplier)
    if escalation is not None:
        qa_tools = qa_tools + build_escalation_tool(escalation)
    builder.add_node("rag_qa", build_qa_agent(model, qa_tools))
    builder.add_node("clarify", clarify_node)
    builder.add_node("draft", lambda state: draft_node(state, supplier))
    # Terminal human-handoff branch — reached when intent routing returns "escalate".
    builder.add_node("escalate", lambda state: escalate_node(state, escalation))

    builder.add_edge(START, "redact")
    builder.add_edge("redact", "intent")
    builder.add_conditional_edges(
        "intent",
        _route_by_intent,
        {"rag_qa": "rag_qa", "parse": "parse", "escalate": "escalate"},
    )
    builder.add_edge("parse", "resolve")
    builder.add_edge("resolve", "add_companions")
    builder.add_edge("add_companions", "check_inventory")
    builder.add_edge("check_inventory", "validate")
    builder.add_edge("validate", "apply")
    builder.add_conditional_edges(
        "apply", _gate_decision, {"clarify": "clarify", "draft": "draft"}
    )
    builder.add_edge("rag_qa", END)
    builder.add_edge("clarify", END)
    builder.add_edge("draft", END)
    builder.add_edge("escalate", END)

    return builder.compile(checkpointer=checkpointer)


def _route_by_intent(state: OrderState) -> str:
    intent = state.get("intent")
    if intent is Intent.ESCALATE:
        return "escalate"
    return "rag_qa" if intent is Intent.QUESTION else "parse"


def _gate_decision(state: OrderState) -> str:
    return gate([op.item for op in state.get("cart_ops", [])])