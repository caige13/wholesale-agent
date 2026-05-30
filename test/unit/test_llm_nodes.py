"""LLM node contract tests — assert state shape & routing, not exact model text.

The nodes only call methods on the injected model, so a fake model keeps these
deterministic and keyless. Exact phrasing/quality of real model output is the
eval set's job, not these.
"""

from types import SimpleNamespace

from src.app.graph.llm_nodes import (
    IntentResult,
    ParsedItem,
    ParsedOrder,
    intent_node,
    parse_node,
    rag_qa_node,
)
from src.domain.models import Intent


class FakeStructuredModel:
    """Stands in for a chat model used via with_structured_output(...).invoke()."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _schema):
        return SimpleNamespace(invoke=lambda _prompt: self._result)


class FakeChatModel:
    """Stands in for a chat model called via .invoke() returning a message."""

    def __init__(self, content):
        self._content = content

    def invoke(self, _prompt):
        return SimpleNamespace(content=self._content)


class FakeCatalog:
    def find_candidates(self, query, k=5):
        return []


def test_intent_node_routes_to_the_classified_intent():
    model = FakeStructuredModel(IntentResult(intent=Intent.QUESTION))
    assert intent_node({"clean_message": "how many per case?"}, model)["intent"] == Intent.QUESTION


def test_parse_node_emits_line_items_from_the_parsed_order():
    model = FakeStructuredModel(ParsedOrder(items=[ParsedItem(phrase="16oz deli", quantity=3)]))
    out = parse_node({"clean_message": "3 cases of 16oz deli"}, model)
    assert out["line_items"][0].raw_text == "16oz deli"
    assert out["line_items"][0].quantity == 3


def test_parse_node_maps_unit_quantity_for_unit_orders():
    model = FakeStructuredModel(ParsedOrder(items=[ParsedItem(phrase="deli", unit_quantity=1200)]))
    out = parse_node({"clean_message": "1200 deli containers"}, model)
    assert out["line_items"][0].unit_quantity == 1200


def test_rag_qa_node_returns_an_answer_without_touching_the_cart():
    model = FakeChatModel("Each case has 500 units.")
    out = rag_qa_node({"clean_message": "how many per case?"}, model, FakeCatalog())
    assert out["answer"] == "Each case has 500 units."
    assert "draft_cart" not in out  # the question path must not mutate the cart