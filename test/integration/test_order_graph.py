"""Order graph wiring — end-to-end control flow (integration, fake LLM + fake catalog).

Exercises the real LangGraph StateGraph deterministically: a scripted model
stands in for Gemini and an in-memory catalog stands in for FAISS, so we assert
routing and terminal state (drafted vs. clarification vs. answer) without real
model calls. Skipped when langgraph isn't installed.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph")

from src.app.graph.agent import LangGraphOrderAgent  # noqa: E402
from src.app.graph.graph import build_graph  # noqa: E402
from src.app.graph.llm_nodes import IntentResult, ParsedItem, ParsedOrder  # noqa: E402
from src.domain.cart import Cart  # noqa: E402
from src.domain.models import CatalogItem, Intent, LineItem, ResolutionCandidate  # noqa: E402
from src.ports.order_agent import AgentResult  # noqa: E402

S = "acme-foodservice"


class ScriptedModel:
    """Fake chat model: canned intent + parse results, and a canned QA answer."""

    def __init__(self, intent, parsed=None, answer=""):
        self._intent = intent
        self._parsed = parsed or ParsedOrder()
        self._answer = answer

    def with_structured_output(self, schema):
        result = IntentResult(intent=self._intent) if schema is IntentResult else self._parsed
        return SimpleNamespace(invoke=lambda _prompt: result)

    def invoke(self, _prompt):
        return SimpleNamespace(content=self._answer)


class FakeCatalog:
    def __init__(self, items_by_sku=None, candidates_by_phrase=None):
        self._items = items_by_sku or {}
        self._candidates = candidates_by_phrase or {}

    def get(self, sku):
        return self._items.get(sku)

    def all(self):
        return list(self._items.values())

    def find_candidates(self, query, k=5):
        return self._candidates.get(query.strip().lower(), [])


def _item(sku, name, **kw):
    return CatalogItem(
        sku=sku, product_name=name, category="x", unit_size="x", case_pack=100,
        supplier=S, **kw,
    )


def _agent(intent, parsed=None, answer="", catalog=None):
    graph = build_graph(ScriptedModel(intent, parsed, answer), catalog or FakeCatalog())
    return LangGraphOrderAgent(graph)


def test_run_returns_an_agent_result():
    assert isinstance(_agent(Intent.ORDER).run("hi", Cart()), AgentResult)


def test_clean_order_drafts_with_items_and_no_clarification():
    straw = _item("STRAW-WRAP", "Wrapped Straws", requires_lids=False, min_order=1)
    catalog = FakeCatalog(
        items_by_sku={"STRAW-WRAP": straw},
        candidates_by_phrase={"wrapped straws": [ResolutionCandidate(item=straw, score=0.95)]},
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="wrapped straws", quantity=2)])
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("2 cases of straws", Cart())
    assert not result.draft_cart.is_empty()
    assert result.clarifications == []


def test_ambiguous_order_asks_for_clarification():
    d8 = _item("DELI-08", "8oz Deli Container")
    d16 = _item("DELI-16", "16oz Deli Container")
    catalog = FakeCatalog(
        candidates_by_phrase={
            "deli containers": [
                ResolutionCandidate(item=d8, score=0.84),
                ResolutionCandidate(item=d16, score=0.82),
            ]
        },
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="deli containers")])
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("some deli", Cart())
    assert result.clarifications


def test_question_returns_answer_and_leaves_the_cart_unchanged():
    line = LineItem(sku="DELI-16", product_name="16oz Deli Container", quantity=2)
    cart = Cart(by_supplier={S: [line]})
    agent = _agent(Intent.QUESTION, answer="Each case has 500 units.")
    result = agent.run("how many per case?", cart)
    assert result.answer == "Each case has 500 units."
    assert result.draft_cart == cart