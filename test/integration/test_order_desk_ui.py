"""UI boundary over the real graph — a conversation end-to-end (fakes, no gradio).

This is the integration layer above the unit specs in test_gradio_app.py: it
drives the actual LangGraph StateGraph through the UI seam (``run_turn`` +
``render_cart``), threading the cart and chat history between turns exactly like
the Gradio callback does. Per CLAUDE.md, a scripted model stands in for Gemini
(no real LLM, no cost) and an in-memory catalog for FAISS, so we assert the
rendered slip and chat reflect real graph output without model calls. Each turn
is a fresh single-turn graph invocation (spec §6); the UI carries state across
turns.

Gradio is never imported — the seam is dependency-light by design.
"""

import pytest

pytest.importorskip("langgraph")

from src.app.graph.agent import LangGraphOrderAgent  # noqa: E402
from src.app.graph.graph import build_graph  # noqa: E402
from src.app.graph.llm_nodes import ParsedItem, ParsedOrder  # noqa: E402
from src.domain.cart import Cart  # noqa: E402
from src.domain.models import Intent, ResolutionCandidate  # noqa: E402
from src.interfaces.gradio_app import render_cart, run_turn  # noqa: E402
from test.fakes import FakeCatalog, FakeSupplier, ScriptedModel  # noqa: E402
from test.fakes import catalog_item as _item  # noqa: E402

S = "acme-foodservice"


def _agent(intent, parsed=None, answer="", catalog=None, supplier=None):
    graph = build_graph(
        ScriptedModel(intent, parsed, answer),
        catalog or FakeCatalog(),
        supplier or FakeSupplier(),
    )
    return LangGraphOrderAgent(graph)


def _deli_catalog():
    deli = _item("DELI-16", "16oz Deli Container", min_order=1)
    return FakeCatalog(
        items_by_sku={"DELI-16": deli},
        candidates_by_phrase={"16oz deli": [ResolutionCandidate(item=deli, score=0.95)]},
    )


def test_an_order_turn_puts_the_resolved_line_on_the_slip():
    parsed = ParsedOrder(items=[ParsedItem(phrase="16oz deli", quantity=3)])
    agent = _agent(Intent.ORDER, parsed=parsed, catalog=_deli_catalog())

    history, cart = run_turn(agent, "3 cases of 16oz deli", [], Cart())

    slip = render_cart(cart)
    assert "16oz Deli Container" in slip
    assert S in slip  # grouped under its supplier
    assert history[0] == {"role": "user", "content": "3 cases of 16oz deli"}
    assert history[-1]["role"] == "assistant"


def test_an_interleaved_question_answers_without_blanking_the_slip():
    # Turn 1: build a cart. Turn 2: ask a question — §11 end-to-end: the slip must
    # still show the line, and the chat must carry both turns.
    parsed = ParsedOrder(items=[ParsedItem(phrase="16oz deli", quantity=3)])
    history, cart = run_turn(
        _agent(Intent.ORDER, parsed=parsed, catalog=_deli_catalog()),
        "3 cases of 16oz deli",
        [],
        cart=Cart(),
    )

    qa_agent = _agent(Intent.QUESTION, answer="Each case has 100 units.", catalog=_deli_catalog())
    history, cart = run_turn(qa_agent, "how many per case?", history, cart)

    assert history[-1] == {"role": "assistant", "content": "Each case has 100 units."}
    assert "16oz Deli Container" in render_cart(cart)  # slip not blanked
    assert len(history) == 4  # two full turns threaded through the UI


def test_an_ambiguous_order_surfaces_a_clarifying_question_in_chat():
    d8 = _item("DELI-08", "8oz Deli Container")
    d16 = _item("DELI-16", "16oz Deli Container")
    catalog = FakeCatalog(
        candidates_by_phrase={
            "deli": [
                ResolutionCandidate(item=d8, score=0.84),
                ResolutionCandidate(item=d16, score=0.82),
            ]
        },
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="deli")])
    agent = _agent(Intent.ORDER, parsed=parsed, catalog=catalog)

    history, cart = run_turn(agent, "some deli containers", [], Cart())

    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"].strip()  # a question surfaced, not a blank bubble
    assert "Awaiting Order" in render_cart(cart)  # nothing resolved onto the slip