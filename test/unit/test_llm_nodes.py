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
)
from src.domain.cart import Cart
from src.domain.models import CartOpKind, Intent
from test.fakes import FakeCatalog


class FakeStructuredModel:
    """Stands in for a chat model used via with_structured_output(...).invoke()."""

    def __init__(self, result):
        self._result = result
        self.prompt = None  # the last prompt it was invoked with

    def with_structured_output(self, _schema):
        def _invoke(prompt):
            self.prompt = prompt
            return self._result

        return SimpleNamespace(invoke=_invoke)


def test_intent_node_routes_to_the_classified_intent():
    model = FakeStructuredModel(IntentResult(intent=Intent.QUESTION))
    assert intent_node({"clean_message": "how many per case?"}, model)["intent"] == Intent.QUESTION


def test_parse_node_emits_an_add_op_with_the_line_item():
    model = FakeStructuredModel(ParsedOrder(items=[ParsedItem(phrase="16oz deli", quantity=3)]))
    out = parse_node(
        {"clean_message": "3 cases of 16oz deli", "draft_cart": Cart()}, model, FakeCatalog()
    )
    op = out["cart_ops"][0]
    assert op.op is CartOpKind.ADD
    assert op.item.raw_text == "16oz deli"
    assert op.item.quantity == 3


def test_parse_node_maps_unit_quantity_for_unit_orders():
    model = FakeStructuredModel(ParsedOrder(items=[ParsedItem(phrase="deli", unit_quantity=1200)]))
    out = parse_node(
        {"clean_message": "1200 deli containers", "draft_cart": Cart()}, model, FakeCatalog()
    )
    assert out["cart_ops"][0].item.unit_quantity == 1200


def test_parse_node_carries_the_classified_op_kind():
    model = FakeStructuredModel(
        ParsedOrder(items=[ParsedItem(phrase="limes", action=CartOpKind.REMOVE)])
    )
    out = parse_node(
        {"clean_message": "drop the limes", "draft_cart": Cart()}, model, FakeCatalog()
    )
    assert out["cart_ops"][0].op is CartOpKind.REMOVE


def test_parse_node_includes_recent_history_so_follow_ups_have_context():
    model = FakeStructuredModel(ParsedOrder(items=[]))
    state = {
        "clean_message": "16oz",
        "draft_cart": Cart(),
        "history": [{"role": "assistant", "content": "which deli size did you mean?"}],
    }
    parse_node(state, model, FakeCatalog())
    assert "which deli size did you mean?" in model.prompt