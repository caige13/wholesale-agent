r"""build_qa_agent — the QUESTION-path tool-calling loop, compiled as a subgraph.

The order path is a fixed deterministic pipeline, but a *question* ("is the 16oz deli
in stock and how much is it?") doesn't fit one shape — so the QA path is a real
LangGraph tool-calling agent: the model is bound to the read-only tools, decides which
to call, a ``ToolNode`` runs them, and the result returns as a ``ToolMessage`` until the
model stops calling tools and answers.

    START -> seed -> assistant --(tool_calls)--> tools --> assistant ...
                              \--(done)--------> finalize -> END

This compiled graph is added to the root graph **as the ``rag_qa`` node itself** (not
called via a blocking ``invoke`` inside a wrapper). ``QAState`` shares ``clean_message``
(in) and ``answer`` (out) with the parent ``OrderState``; the chat ``messages`` channel
is private to this subgraph, so the deterministic order state never grows a message
log. Because it's embedded as a node, ``parent.stream(stream_mode="messages",
subgraphs=True)`` streams the answer tokens for free — no nested stream plumbing.

The tool loop counts against the *parent's* ``recursion_limit`` (default 25), which
comfortably bounds a few tool round-trips.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class QAState(TypedDict, total=False):
    clean_message: str  # shared IN with OrderState — the redacted question
    history: list[dict]  # shared IN with OrderState — recent turns for follow-up context
    messages: Annotated[list, add_messages]  # private to the loop (chat history)
    answer: str  # shared OUT with OrderState — the final text answer


_SYSTEM_PROMPT = SystemMessage(
    "You are the whole sale supplier's order desk's product assistant. Answer the "
    "restaurant's question using ONLY the tools and their results — do not rely on prior "
    "knowledge or invent details.\n"
    "- Use search_catalog to find a product and its SKU from a description.\n"
    "- Use check_inventory for live stock and get_price for the per-case price "
    "(neither lives in the catalog).\n"
    "If the question is outside product stock/pricing (returns, billing, account changes, "
    "complaints) or the catalog and tools simply can't answer it, call escalate_to_human "
    "with a brief reason instead of guessing — then give the customer the ticket reference "
    "from the tool result so they can track it. Be concise."
)


def build_qa_agent(model, tools):
    """Compile the seed -> (assistant <-> tools) -> finalize loop, bound to the tools."""
    bound = model.bind_tools(tools)

    def seed(state: QAState) -> dict:
        # Seed the loop with the recent conversation so a follow-up ("16") resolves
        # against the prior question, then the current message.
        prior = [
            AIMessage(turn.get("content", "")) if turn.get("role") == "assistant"
            else HumanMessage(turn.get("content", ""))
            for turn in (state.get("history") or [])[-6:]
        ]
        return {"messages": [*prior, HumanMessage(state["clean_message"])]}

    def assistant(state: QAState) -> dict:
        return {"messages": [bound.invoke([_SYSTEM_PROMPT, *state["messages"]])]}

    def finalize(state: QAState) -> dict:
        return {"answer": _message_text(state["messages"][-1])}

    builder = StateGraph(QAState)
    builder.add_node("seed", seed)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "assistant")
    builder.add_conditional_edges(
        "assistant", tools_condition, {"tools": "tools", END: "finalize"}
    )
    builder.add_edge("tools", "assistant")
    builder.add_edge("finalize", END)
    return builder.compile()


def _message_text(message: object) -> str:
    """Flatten a chat message's content to text. Gemini returns a list of content
    blocks (e.g. ``[{"type": "text", "text": "..."}]``); others return a string.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts)
    return str(content)