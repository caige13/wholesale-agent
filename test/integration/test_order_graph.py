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
from test.fakes import FakeCatalog, FakeEscalation, FakeSupplier, ScriptedModel  # noqa: E402
from test.fakes import catalog_item as _item  # noqa: E402

SUPPLIER = "acme-foodservice"


def _agent(intent, parsed=None, answer="", catalog=None, supplier=None, tool_steps=None,
           escalation=None):
    graph = build_graph(
        ScriptedModel(intent, parsed, answer, tool_steps),
        catalog or FakeCatalog(),
        supplier or FakeSupplier(),
        escalation=escalation,
    )
    return LangGraphOrderAgent(graph)


def test_run_returns_an_agent_result():
    assert isinstance(_agent(Intent.ORDER).run("hi", Cart()), AgentResult)


def test_a_checkpointer_persists_the_cart_across_turns_on_one_thread():
    # With a checkpointer the running cart is restored from thread state: turn 2 omits
    # the cart entirely, yet its add stacks onto turn 1's persisted line (it would be 2,
    # not 4, if nothing had persisted). A second thread starts clean — state is isolated.
    from langgraph.checkpoint.memory import MemorySaver

    straw = _item("STRAW-WRAP", "Wrapped Straws", min_order=1)
    catalog = FakeCatalog(
        items_by_sku={"STRAW-WRAP": straw},
        candidates_by_phrase={"wrapped straws": [ResolutionCandidate(item=straw, score=0.95)]},
    )
    graph = build_graph(
        ScriptedModel(
            Intent.ORDER, ParsedOrder(items=[ParsedItem(phrase="wrapped straws", quantity=2)])
        ),
        catalog,
        FakeSupplier(),
        checkpointer=MemorySaver(),
    )
    agent = LangGraphOrderAgent(graph)

    first = agent.run("2 cases of wrapped straws", Cart(), thread_id="t1")
    assert first.draft_cart.all_lines()[0].quantity == 2

    second = agent.run("2 more cases of wrapped straws", thread_id="t1")  # cart omitted
    assert second.draft_cart.all_lines()[0].quantity == 4  # restored, then stacked

    other = agent.run("2 cases of wrapped straws", Cart(), thread_id="t2")
    assert other.draft_cart.all_lines()[0].quantity == 2  # isolated per thread_id


def test_a_prior_question_answer_does_not_leak_into_a_later_order_turn():
    # Under a checkpointer, per-turn outputs persist; a question's `answer` must not
    # survive into a later order turn (it used to surface as the order's reply).
    from langgraph.checkpoint.memory import MemorySaver

    straw = _item("STRAW-WRAP", "Wrapped Straws", min_order=1)
    catalog = FakeCatalog(
        items_by_sku={"STRAW-WRAP": straw},
        candidates_by_phrase={"wrapped straws": [ResolutionCandidate(item=straw, score=0.95)]},
    )
    graph = build_graph(
        ScriptedModel(
            Intent.ORDER, ParsedOrder(items=[ParsedItem(phrase="wrapped straws", quantity=2)])
        ),
        catalog,
        FakeSupplier(),
        checkpointer=MemorySaver(),
    )
    agent = LangGraphOrderAgent(graph)

    # Simulate a prior question turn having left an answer in this thread's state.
    graph_config = {"configurable": {"thread_id": "t1"}}
    graph.update_state(graph_config, {"answer": "We have 120 cases of DELI-32 on hand."})

    result = agent.run("2 cases of wrapped straws", Cart(), thread_id="t1")
    assert result.answer is None  # the order turn must not inherit the stale answer
    assert [li.sku for li in result.draft_cart.all_lines()] == ["STRAW-WRAP"]


def test_record_turn_persists_history_to_the_checkpointer():
    # The UI doesn't thread history anymore — the agent records each turn into its own
    # checkpointed state, so the next turn reads it back.
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph(
        ScriptedModel(Intent.ORDER), FakeCatalog(), FakeSupplier(), checkpointer=MemorySaver()
    )
    agent = LangGraphOrderAgent(graph)
    agent.run("hi", Cart(), thread_id="t1")
    agent.record_turn("hi", "hello there", thread_id="t1")
    history = graph.get_state({"configurable": {"thread_id": "t1"}}).values["history"]
    assert {"role": "user", "content": "hi"} in history
    assert {"role": "assistant", "content": "hello there"} in history


def test_record_turn_is_a_noop_without_a_checkpointer():
    _agent(Intent.ORDER).record_turn("hi", "hello", thread_id="t1")  # must not raise


def test_record_turn_redacts_pii_before_persisting_history():
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph(
        ScriptedModel(Intent.ORDER), FakeCatalog(), FakeSupplier(), checkpointer=MemorySaver()
    )
    agent = LangGraphOrderAgent(graph)
    agent.record_turn("call me at 555-123-4567", "sure", thread_id="t1")
    history = graph.get_state({"configurable": {"thread_id": "t1"}}).values["history"]
    user_msg = next(h["content"] for h in history if h["role"] == "user")
    assert "555-123-4567" not in user_msg  # raw PII never lands in persisted history


def test_stream_run_emits_progress_events_then_a_final_result():
    straw = _item("STRAW-WRAP", "Wrapped Straws", min_order=1)
    catalog = FakeCatalog(
        items_by_sku={"STRAW-WRAP": straw},
        candidates_by_phrase={"wrapped straws": [ResolutionCandidate(item=straw, score=0.95)]},
    )
    agent = _agent(
        Intent.ORDER,
        parsed=ParsedOrder(items=[ParsedItem(phrase="wrapped straws", quantity=2)]),
        catalog=catalog,
    )
    events = list(agent.stream_run("2 cases of wrapped straws", Cart()))
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "result"  # the stream always ends with the finished turn
    progressed = {payload for kind, payload in events if kind == "progress"}
    assert {"redact", "intent", "parse", "apply"} <= progressed  # real graph nodes, as they run
    result = events[-1][1]
    assert [li.sku for li in result.draft_cart.all_lines()] == ["STRAW-WRAP"]


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


def test_a_clean_order_builds_a_draft_but_does_not_confirm_until_placed():
    # Adding an item never auto-submits — the cart is a running draft (the bug:
    # it used to confirm every clean turn, so "I wasn't done" got a second order).
    straw = _item("STRAW-WRAP", "Wrapped Straws", min_order=1)
    catalog = FakeCatalog(
        items_by_sku={"STRAW-WRAP": straw},
        candidates_by_phrase={"wrapped straws": [ResolutionCandidate(item=straw, score=0.95)]},
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="wrapped straws", quantity=2)])  # no checkout
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("2 cases of straws", Cart())
    assert [li.sku for li in result.draft_cart.all_lines()] == ["STRAW-WRAP"]  # cart built
    assert result.confirmation is None  # but NOT submitted


def test_placing_the_order_fills_unit_price_and_confirms():
    straw = _item("STRAW-WRAP", "Wrapped Straws", min_order=1)
    catalog = FakeCatalog(
        items_by_sku={"STRAW-WRAP": straw},
        candidates_by_phrase={"wrapped straws": [ResolutionCandidate(item=straw, score=0.95)]},
    )
    # "2 cases of straws, that's it" — adds the item AND places the order.
    parsed = ParsedOrder(
        items=[ParsedItem(phrase="wrapped straws", quantity=2)], place_order=True
    )
    supplier = FakeSupplier(prices={"STRAW-WRAP": 16.60})
    result = _agent(
        Intent.ORDER, parsed=parsed, catalog=catalog, supplier=supplier
    ).run("2 cases of straws, that's it", Cart())
    assert result.clarifications == []
    assert result.draft_cart.all_lines()[0].unit_price == 16.60
    assert result.confirmation is not None
    assert result.confirmation.order_id == "TEST-ORDER"


def test_placing_the_order_with_no_new_items_submits_the_existing_cart():
    existing = LineItem(
        sku="STRAW-WRAP", product_name="Wrapped Straws", supplier=SUPPLIER, quantity=2
    )
    cart = Cart(by_supplier={SUPPLIER: [existing]})
    result = _agent(Intent.ORDER, parsed=ParsedOrder(place_order=True)).run("place the order", cart)
    assert result.confirmation is not None  # the running draft is submitted on checkout
    assert [li.sku for li in result.draft_cart.all_lines()] == ["STRAW-WRAP"]  # unchanged


def test_ambiguous_order_asks_for_clarification():
    deli_8oz = _item("DELI-08", "8oz Deli Container")
    deli_16oz = _item("DELI-16", "16oz Deli Container")
    catalog = FakeCatalog(
        candidates_by_phrase={
            "deli containers": [
                ResolutionCandidate(item=deli_8oz, score=0.84),
                ResolutionCandidate(item=deli_16oz, score=0.82),
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
    existing = LineItem(
        sku="DELI-16", product_name="16oz Deli Container", supplier=SUPPLIER, quantity=2
    )
    cart = Cart(by_supplier={SUPPLIER: [existing]})
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("make it 3", cart)
    lines = result.draft_cart.by_supplier[SUPPLIER]
    assert len(lines) == 1  # replaced, not duplicated
    assert lines[0].quantity == 3


def test_remove_drops_an_existing_cart_line():
    lime = _item("LIME-FRESH", "Fresh Limes", aliases=["limes"])
    catalog = FakeCatalog(
        items_by_sku={"LIME-FRESH": lime},
        candidates_by_phrase={"limes": [ResolutionCandidate(item=lime, score=0.95)]},
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="limes", action=CartOpKind.REMOVE)])
    existing = LineItem(sku="LIME-FRESH", product_name="Fresh Limes", supplier=SUPPLIER, quantity=1)
    cart = Cart(by_supplier={SUPPLIER: [existing]})
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("drop the limes", cart)
    assert result.draft_cart.is_empty()


def test_question_returns_answer_and_leaves_the_cart_unchanged():
    line = LineItem(sku="DELI-16", product_name="16oz Deli Container", quantity=2)
    cart = Cart(by_supplier={SUPPLIER: [line]})
    agent = _agent(Intent.QUESTION, answer="Each case has 500 units.")
    result = agent.run("how many per case?", cart)
    assert result.answer == "Each case has 500 units."
    assert result.draft_cart == cart


def test_question_path_runs_a_tool_call_loop_then_answers():
    # The QUESTION branch is a real tool-calling subgraph: the model calls search_catalog,
    # the ToolNode runs it, then the model answers — all read-only, cart untouched.
    deli = _item("DELI-16", "16oz Deli Container")
    catalog = FakeCatalog(
        candidates_by_phrase={"16oz deli": [ResolutionCandidate(item=deli, score=0.95)]}
    )
    agent = _agent(
        Intent.QUESTION,
        catalog=catalog,
        tool_steps=[
            [{"name": "search_catalog", "args": {"query": "16oz deli"}}],
            "The 16oz deli container comes 500 per case.",
        ],
    )
    cart = Cart(by_supplier={
        SUPPLIER: [LineItem(sku="DELI-16", product_name="16oz Deli", quantity=2)]
    })
    result = agent.run("how many 16oz deli per case?", cart)
    assert result.answer == "The 16oz deli container comes 500 per case."
    assert result.draft_cart == cart  # question path leaves the cart intact


# --- escalation: hand off to a human ------------------------------------------
def test_explicit_escalation_hands_off_to_a_human_and_leaves_the_cart_untouched():
    # An ESCALATE-classified turn opens a handoff ticket and ends — no order edit.
    line = LineItem(
        sku="DELI-16", product_name="16oz Deli Container", supplier=SUPPLIER, quantity=2
    )
    cart = Cart(by_supplier={SUPPLIER: [line]})
    agent = _agent(Intent.ESCALATE, escalation=FakeEscalation())
    result = agent.run("I need to dispute an invoice — can I talk to a person?", cart)
    assert result.handoff is not None
    assert result.handoff.ticket_id == "TEST-HANDOFF"
    assert result.draft_cart == cart  # escalation never touches the order
    assert result.clarifications == []  # it hands off rather than asking the desk's questions


# --- companion add-on flow: accept a pending offer, add it by SKU --------------
def _deli32():
    return CatalogItem(
        sku="DELI-32", product_name="32oz Deli Container", category="containers",
        unit_size="32oz", case_pack=480, supplier=SUPPLIER, companion_skus=["LID-DELI"],
    )


def _lid_deli():
    return CatalogItem(
        sku="LID-DELI", product_name="Deli Container Lid", category="lids",
        unit_size="fits 8-32oz", case_pack=500, supplier=SUPPLIER,
    )


def _deli16():
    return CatalogItem(
        sku="DELI-16", product_name="16oz Deli Container", category="containers",
        unit_size="16oz", case_pack=500, supplier=SUPPLIER, companion_skus=["LID-DELI"],
    )


def _cart_with_pending_lid(quantity=3):
    parent = LineItem(
        sku="DELI-32", product_name="32oz Deli Container", supplier=SUPPLIER, quantity=quantity,
        flags=[Flag.NEEDS_COMPANION],
        companions=[Companion(sku="LID-DELI", product_name="Deli Container Lid")],
    )
    return Cart(by_supplier={SUPPLIER: [parent]})


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


def test_adding_an_item_without_a_quantity_asks_how_many_and_does_not_draft():
    cup = _item("PCUP-2", "2oz Portion Cup", aliases=["salsa cups"])
    catalog = FakeCatalog(
        items_by_sku={"PCUP-2": cup},
        candidates_by_phrase={"salsa cups": [ResolutionCandidate(item=cup, score=0.95)]},
    )
    parsed = ParsedOrder(items=[ParsedItem(phrase="salsa cups")])  # resolves, but no quantity
    result = _agent(Intent.ORDER, parsed=parsed, catalog=catalog).run("I want salsa cups", Cart())
    assert result.clarifications  # asks "how many?"
    assert result.draft_cart.is_empty()  # no quantity-less line landed
    assert result.confirmation is None  # and it did not draft/confirm


def test_adding_a_second_deli_size_re_offers_and_tops_up_the_shared_lid():
    catalog = FakeCatalog(
        items_by_sku={"DELI-32": _deli32(), "DELI-16": _deli16(), "LID-DELI": _lid_deli()},
        candidates_by_phrase={"16oz deli": [ResolutionCandidate(item=_deli16(), score=0.95)]},
    )
    # Cart already covers 3× 32oz deli (1440 units) with exactly 3 lid cases.
    cart = Cart(by_supplier={SUPPLIER: [
        LineItem(sku="DELI-32", product_name="32oz Deli Container", supplier=SUPPLIER, quantity=3),
        LineItem(sku="LID-DELI", product_name="Deli Container Lid", supplier=SUPPLIER, quantity=3),
    ]})
    # Turn 1: add 2× 16oz deli → 2440 units now need 5 lid cases, only 3 present → re-offer.
    add = ParsedOrder(items=[ParsedItem(phrase="16oz deli", quantity=2)])
    turn1 = _agent(Intent.ORDER, parsed=add, catalog=catalog).run("add 2 cases of 16oz deli", cart)
    assert turn1.clarifications  # the shared lid is re-offered for the added size
    assert next(li for li in turn1.draft_cart.all_lines() if li.sku == "LID-DELI").quantity == 3
    # Turn 2: "yes" → lid set to cover ALL delis: ceil(2440 / 500) = 5; no further offer.
    accept = ParsedOrder(accepted_companions=[AcceptedCompanion(name="Deli Container Lid")])
    turn2 = _agent(Intent.ORDER, parsed=accept, catalog=catalog).run("yes", turn1.draft_cart)
    assert next(li for li in turn2.draft_cart.all_lines() if li.sku == "LID-DELI").quantity == 5
    assert turn2.clarifications == []