"""Order graph wiring — end-to-end control flow (integration, fake LLM + fake catalog).

Exercises the real LangGraph StateGraph deterministically: a scripted model
stands in for Gemini and an in-memory catalog stands in for FAISS, so we assert
routing and terminal state (drafted vs. clarification vs. answer) without real
model calls. Skipped when langgraph isn't installed.
"""

import pytest

pytest.importorskip("langgraph")

from src.app.graph.agent import LangGraphOrderAgent  # noqa: E402
from src.app.graph.graph import build_graph  # noqa: E402
from src.app.graph.llm_nodes import AcceptedCompanion, ParsedItem, ParsedOrder  # noqa: E402
from src.domain.cart import Cart  # noqa: E402
from src.domain.models import (  # noqa: E402
    CartOpKind,
    CatalogItem,
    Companion,
    Flag,
    Intent,
    LineItem,
    ResolutionCandidate,
)
from src.ports.order_agent import AgentResult  # noqa: E402
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


def test_run_returns_an_agent_result():
    assert isinstance(_agent(Intent.ORDER).run("hi", Cart()), AgentResult)


def test_clean_order_drafts_with_items_and_no_clarification():
    straw = _item("STRAW-WRAP", "Wrapped Straws", min_order=1)
    catalog = FakeCatalog(
        items_by_sku={"STRAW-WRAP": straw},
        candidates_by_phrase={"wrapped straws": [ResolutionCandidate(item=straw, score=0.95)]},
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="wrapped straws", quantity=2)])
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("2 cases of straws", Cart())
    assert not result.draft_cart.is_empty()
    assert result.clarifications == []


def test_out_of_stock_order_drafts_the_line_but_asks_to_clarify():
    lime = _item("LIME-FRESH", "Fresh Limes", aliases=["limes"])
    catalog = FakeCatalog(
        items_by_sku={"LIME-FRESH": lime},
        candidates_by_phrase={"limes": [ResolutionCandidate(item=lime, score=0.95)]},
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="limes", quantity=2)])
    supplier = FakeSupplier(out_of_stock={"LIME-FRESH"})
    result = _agent(
        Intent.ORDER, parsed=parsed, catalog=catalog, supplier=supplier
    ).run("2 cases of limes", Cart())
    assert result.clarifications  # gate stops on OUT_OF_STOCK
    # The line still lands in the cart (apply runs before the gate).
    assert [li.sku for li in result.draft_cart.all_lines()] == ["LIME-FRESH"]


def test_clean_order_fills_unit_price_and_confirms_the_order():
    straw = _item("STRAW-WRAP", "Wrapped Straws", min_order=1)
    catalog = FakeCatalog(
        items_by_sku={"STRAW-WRAP": straw},
        candidates_by_phrase={"wrapped straws": [ResolutionCandidate(item=straw, score=0.95)]},
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="wrapped straws", quantity=2)])
    supplier = FakeSupplier(prices={"STRAW-WRAP": 16.60})
    result = _agent(
        Intent.ORDER, parsed=parsed, catalog=catalog, supplier=supplier
    ).run("2 cases of straws", Cart())
    assert result.clarifications == []
    assert result.draft_cart.all_lines()[0].unit_price == 16.60
    assert result.confirmation is not None
    assert result.confirmation.order_id == "TEST-ORDER"


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


def test_set_quantity_updates_an_existing_cart_line_in_place():
    deli = _item("DELI-16", "16oz Deli Container", aliases=["16oz deli"])
    catalog = FakeCatalog(
        items_by_sku={"DELI-16": deli},
        candidates_by_phrase={"16oz deli": [ResolutionCandidate(item=deli, score=0.95)]},
    )
    parsed = ParsedOrder(
        items=[ParsedItem(phrase="16oz deli", quantity=3, action=CartOpKind.SET_QUANTITY)]
    )
    existing = LineItem(sku="DELI-16", product_name="16oz Deli Container", supplier=S, quantity=2)
    cart = Cart(by_supplier={S: [existing]})
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("make it 3", cart)
    lines = result.draft_cart.by_supplier[S]
    assert len(lines) == 1  # replaced, not duplicated
    assert lines[0].quantity == 3


def test_remove_drops_an_existing_cart_line():
    lime = _item("LIME-FRESH", "Fresh Limes", aliases=["limes"])
    catalog = FakeCatalog(
        items_by_sku={"LIME-FRESH": lime},
        candidates_by_phrase={"limes": [ResolutionCandidate(item=lime, score=0.95)]},
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="limes", action=CartOpKind.REMOVE)])
    existing = LineItem(sku="LIME-FRESH", product_name="Fresh Limes", supplier=S, quantity=1)
    cart = Cart(by_supplier={S: [existing]})
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("drop the limes", cart)
    assert result.draft_cart.is_empty()


def test_question_returns_answer_and_leaves_the_cart_unchanged():
    line = LineItem(sku="DELI-16", product_name="16oz Deli Container", quantity=2)
    cart = Cart(by_supplier={S: [line]})
    agent = _agent(Intent.QUESTION, answer="Each case has 500 units.")
    result = agent.run("how many per case?", cart)
    assert result.answer == "Each case has 500 units."
    assert result.draft_cart == cart


# --- companion add-on flow: accept a pending offer, add it by SKU --------------
def _deli32():
    return CatalogItem(
        sku="DELI-32", product_name="32oz Deli Container", category="containers",
        unit_size="32oz", case_pack=480, supplier=S, companion_skus=["LID-DELI"],
    )


def _lid_deli():
    return CatalogItem(
        sku="LID-DELI", product_name="Deli Container Lid", category="lids",
        unit_size="fits 8-32oz", case_pack=500, supplier=S,
    )


def _cart_with_pending_lid(quantity=3):
    parent = LineItem(
        sku="DELI-32", product_name="32oz Deli Container", supplier=S, quantity=quantity,
        flags=[Flag.NEEDS_COMPANION],
        companions=[Companion(sku="LID-DELI", product_name="Deli Container Lid")],
    )
    return Cart(by_supplier={S: [parent]})


def test_accepting_a_companion_offer_adds_it_by_sku_and_drafts():
    catalog = FakeCatalog(items_by_sku={"DELI-32": _deli32(), "LID-DELI": _lid_deli()})
    parsed = ParsedOrder(accepted_companions=[AcceptedCompanion(name="Deli Container Lid")])
    offer = "32oz Deli Container needs matching Deli Container Lid — should I add them?"
    history = [
        {"role": "user", "content": "3 cases of 32oz deli containers"},
        {"role": "assistant", "content": offer},
    ]
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run(
        "yes please", _cart_with_pending_lid(), history
    )
    assert sorted(li.sku for li in result.draft_cart.all_lines()) == ["DELI-32", "LID-DELI"]
    lid = next(li for li in result.draft_cart.all_lines() if li.sku == "LID-DELI")
    assert lid.quantity == 3  # ceil(3 * 480 / 500) covers 1440 containers
    assert result.clarifications == []  # the offer is resolved — it drafts, no re-ask


def test_accepting_while_changing_the_parent_quantity_sizes_the_companion_to_the_new_total():
    catalog = FakeCatalog(
        items_by_sku={"DELI-32": _deli32(), "LID-DELI": _lid_deli()},
        candidates_by_phrase={"32oz deli": [ResolutionCandidate(item=_deli32(), score=0.95)]},
    )
    parsed = ParsedOrder(
        items=[ParsedItem(phrase="32oz deli", quantity=6, action=CartOpKind.SET_QUANTITY)],
        accepted_companions=[AcceptedCompanion(name="Deli Container Lid")],
    )
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run(
        "yes, make it 6 cases", _cart_with_pending_lid()
    )
    lid = next(li for li in result.draft_cart.all_lines() if li.sku == "LID-DELI")
    assert lid.quantity == 6  # ceil(6 * 480 / 500) = 6, sized to the new total, not the old 3


def test_declining_or_vague_reply_leaves_the_offer_open_without_adding():
    # No accepted companions and no new items → nothing is added; the cart is unchanged.
    catalog = FakeCatalog(items_by_sku={"DELI-32": _deli32(), "LID-DELI": _lid_deli()})
    result = _agent(Intent.ORDER, parsed=ParsedOrder(), catalog=catalog).run(
        "hmm, maybe later", _cart_with_pending_lid()
    )
    assert [li.sku for li in result.draft_cart.all_lines()] == ["DELI-32"]  # no lid forced on