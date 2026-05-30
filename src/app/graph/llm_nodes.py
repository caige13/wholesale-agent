"""LLM-backed graph nodes — intent classification, order parsing, and RAG Q&A.

Each node takes the injected LangChain chat model (Gemini at runtime, a fake in
tests) and calls it; this module imports no LangChain itself, so the contract
tests stay keyless. The structured-output schemas below are what the model fills.

The probabilistic quality of these nodes (did it parse/classify correctly?) is
the eval set's job — the contract tests here only pin the state shape & routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.domain.models import Intent, LineItem, ResolutionCandidate
from src.ports import CatalogRepository

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# --- Structured-output schemas the model fills ------------------------------
class ParsedItem(BaseModel):
    """One requested product extracted from the message."""

    phrase: str  # the product phrase as the user said it, e.g. "16oz deli"
    quantity: int | None = None  # number of cases, if stated
    unit_quantity: int | None = None  # raw unit count, if stated ("1200 containers")


class ParsedOrder(BaseModel):
    items: list[ParsedItem] = Field(default_factory=list)


class IntentResult(BaseModel):
    intent: Intent


# --- Prompts ----------------------------------------------------------------
_INTENT_INSTRUCTIONS = (
    "Classify the restaurant's message as one of: 'order' (placing/adding items), "
    "'reorder' (repeat a usual order), or 'question' (asking about a product). "
    "Message:\n"
)
_PARSE_INSTRUCTIONS = (
    "Extract the supply items the restaurant wants to order. For each, give the "
    "product phrase as said, and either the number of cases (quantity) or a raw "
    "unit count (unit_quantity) if they ordered in units. Message:\n"
)
_QA_INSTRUCTIONS = (
    "Answer the restaurant's product question using only the catalog context. Be "
    "concise. If the context doesn't cover it, say so.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}"
)


# --- Nodes (graph order: intent -> parse ... and the question branch) -------
def intent_node(state: dict, model: BaseChatModel) -> dict:
    result = model.with_structured_output(IntentResult).invoke(
        _INTENT_INSTRUCTIONS + state["clean_message"]
    )
    return {"intent": result.intent}


def parse_node(state: dict, model: BaseChatModel) -> dict:
    parsed = model.with_structured_output(ParsedOrder).invoke(
        _PARSE_INSTRUCTIONS + state["clean_message"]
    )
    line_items = [
        LineItem(raw_text=item.phrase, quantity=item.quantity, unit_quantity=item.unit_quantity)
        for item in parsed.items
    ]
    return {"line_items": line_items}


def rag_qa_node(state: dict, model: BaseChatModel, catalog: CatalogRepository) -> dict:
    candidates = catalog.find_candidates(state["clean_message"], k=3)
    prompt = _QA_INSTRUCTIONS.format(
        context=_format_context(candidates), question=state["clean_message"]
    )
    message = model.invoke(prompt)
    return {"answer": getattr(message, "content", str(message))}


def _format_context(candidates: list[ResolutionCandidate]) -> str:
    if not candidates:
        return "(no matching catalog entries)"
    return "\n".join(
        f"- {c.item.product_name} ({c.item.unit_size}, {c.item.case_pack} per case)"
        for c in candidates
    )