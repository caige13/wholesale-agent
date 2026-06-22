"""Deterministic graph nodes — thin wrappers over the pure functions.

The per-turn carrier is ``cart_ops``: each node maps OrderState in → a partial
update out, enriching/applying the ops. Keyless (a fake catalog repo stands in
for the FAISS retriever); the LLM nodes are contract-tested separately.
"""

from src.app.graph.nodes import (
    _undercovered_companions,
    add_companions_node,
    apply_node,
    build_clarifications,
    clarify_node,
    draft_node,
    pending_companions,
    redact_node,
    resolve_node,
    validate_node,
)
from src.domain.cart import Cart
from src.domain.models import (
    CartOp,
    CartOpKind,
    Companion,
    Flag,
    LineItem,
    OrderStatus,
    ResolutionCandidate,
)
from test.fakes import FakeCatalog as FakeCatalogRepo
from test.fakes import FakeSupplier

S = "acme-foodservice"


def _add(item: LineItem) -> CartOp:
    return CartOp(op=CartOpKind.ADD, item=item)


def test_redact_node_cleans_the_message_and_flags_pii():
    out = redact_node({"raw_message": "call 555-123-4567 for 16 oz deli"})
    assert "[REDACTED_PHONE]" in out["clean_message"]
    assert "16oz" in out["clean_message"]
    assert "phone" in out["pii_found"]


def test_resolve_node_sets_the_sku_on_each_op_item(make_catalog_item):
    item = make_catalog_item(sku="PCUP-2", product_name="2oz Portion Cup", aliases=["salsa cups"])
    repo = FakeCatalogRepo(
        candidates_by_phrase={"salsa cups": [ResolutionCandidate(item=item, score=0.9)]}
    )
    out = resolve_node({"cart_ops": [_add(LineItem(raw_text="salsa cups"))]}, repo)
    assert out["cart_ops"][0].item.sku == "PCUP-2"


def test_resolve_node_scopes_retrieval_to_the_selected_suppliers(make_catalog_item):
    # Two suppliers carry a "salsa cups" match; supplier-a even scores higher. The
    # customer selected only supplier-b, so a is filtered out before resolution and
    # the line resolves to b's SKU (multi-tenant scoping).
    a = make_catalog_item(sku="PCUP-2", product_name="2oz Portion Cup",
                          aliases=["salsa cups"], supplier="supplier-a")
    b = make_catalog_item(sku="SOUFFLE-2", product_name="2oz Souffle Cup",
                          aliases=["salsa cups"], supplier="supplier-b")
    repo = FakeCatalogRepo(candidates_by_phrase={"salsa cups": [
        ResolutionCandidate(item=a, score=0.95), ResolutionCandidate(item=b, score=0.9),
    ]})
    state = {"cart_ops": [_add(LineItem(raw_text="salsa cups"))],
             "selected_suppliers": ["supplier-b"]}
    out = resolve_node(state, repo)
    assert out["cart_ops"][0].item.sku == "SOUFFLE-2"
    assert out["cart_ops"][0].item.supplier == "supplier-b"


def test_resolve_node_searches_all_suppliers_when_none_are_selected(make_catalog_item):
    # No selection (the default) ⇒ the highest-scoring match across suppliers wins.
    a = make_catalog_item(sku="PCUP-2", aliases=["salsa cups"], supplier="supplier-a")
    b = make_catalog_item(sku="SOUFFLE-2", aliases=["salsa cups"], supplier="supplier-b")
    repo = FakeCatalogRepo(candidates_by_phrase={"salsa cups": [
        ResolutionCandidate(item=a, score=0.95), ResolutionCandidate(item=b, score=0.7),
    ]})
    out = resolve_node({"cart_ops": [_add(LineItem(raw_text="salsa cups"))]}, repo)
    assert out["cart_ops"][0].item.sku == "PCUP-2"


def test_validate_node_raises_companion_flag_and_names_the_offer(make_catalog_item):
    deli = make_catalog_item(sku="DELI-16", companion_skus=["LID-DELI"])
    lid = make_catalog_item(sku="LID-DELI", product_name="Deli Container Lid", category="lids")
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli, "LID-DELI": lid})
    state = {"cart_ops": [_add(LineItem(raw_text="deli", sku="DELI-16", supplier=S, quantity=3))]}
    item = validate_node(state, repo)["cart_ops"][0].item
    assert Flag.NEEDS_COMPANION in item.flags
    assert [c.product_name for c in item.companions] == ["Deli Container Lid"]


def test_validate_node_makes_no_offer_when_the_cart_already_covers_the_companion(make_catalog_item):
    deli = make_catalog_item(sku="DELI-16", case_pack=500, companion_skus=["LID-DELI"])
    lid = make_catalog_item(
        sku="LID-DELI", product_name="Deli Container Lid", category="lids", case_pack=500
    )
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli, "LID-DELI": lid})
    # 3 cases of 16oz deli (1500 units) need 3 lid cases — and 3 are already present.
    cart = Cart(by_supplier={S: [LineItem(sku="LID-DELI", supplier=S, quantity=3)]})
    state = {
        "cart_ops": [_add(LineItem(raw_text="deli", sku="DELI-16", supplier=S, quantity=3))],
        "draft_cart": cart,
    }
    item = validate_node(state, repo)["cart_ops"][0].item
    assert Flag.NEEDS_COMPANION not in item.flags  # lids adequately cover → no nudge


def test_validate_node_does_not_re_ask_for_a_companion_on_a_quantity_change(make_catalog_item):
    # Bumping the quantity of a container already in the cart shouldn't re-nag.
    deli = make_catalog_item(sku="DELI-16", companion_skus=["LID-DELI"])
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli})
    line = LineItem(raw_text="deli", sku="DELI-16", supplier=S, quantity=3)
    out = validate_node({"cart_ops": [CartOp(op=CartOpKind.SET_QUANTITY, item=line)]}, repo)
    assert Flag.NEEDS_COMPANION not in out["cart_ops"][0].item.flags


def test_validate_node_skips_remove_ops(make_catalog_item):
    deli = make_catalog_item(sku="DELI-16", companion_skus=["LID-DELI"])
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli})
    remove = CartOp(op=CartOpKind.REMOVE, item=LineItem(raw_text="deli", sku="DELI-16"))
    out = validate_node({"cart_ops": [remove]}, repo)
    assert out["cart_ops"][0].item.flags == []  # a removal isn't validated for companions etc.


def test_apply_node_adds_resolved_ops_to_the_cart():
    line = LineItem(
        raw_text="deli", sku="DELI-16", product_name="16oz Deli Container",
        supplier=S, quantity=2, confidence=0.9,
    )
    out = apply_node({"cart_ops": [_add(line)], "draft_cart": Cart()})
    assert out["draft_cart"].by_supplier[S][0].sku == "DELI-16"


def test_apply_node_set_quantity_replaces_the_existing_line():
    existing = LineItem(sku="DELI-16", product_name="16oz Deli Container", supplier=S, quantity=2)
    update = existing.model_copy(update={"quantity": 3})
    op = CartOp(op=CartOpKind.SET_QUANTITY, item=update)
    out = apply_node({"cart_ops": [op], "draft_cart": Cart(by_supplier={S: [existing]})})
    assert out["draft_cart"].by_supplier[S][0].quantity == 3


def test_apply_node_remove_drops_the_line():
    lime = LineItem(sku="LIME-FRESH", product_name="Fresh Limes", supplier=S, quantity=1)
    op = CartOp(op=CartOpKind.REMOVE, item=lime)
    out = apply_node({"cart_ops": [op], "draft_cart": Cart(by_supplier={S: [lime]})})
    assert out["draft_cart"].is_empty()


def test_apply_node_skips_ops_whose_item_is_unresolved():
    op = _add(LineItem(raw_text="???", sku=None))
    out = apply_node({"cart_ops": [op], "draft_cart": Cart()})
    assert out["draft_cart"].is_empty()


def test_apply_node_skips_an_add_with_no_quantity():
    # "I want salsa cups" resolves but has no amount — don't land a half-line; the
    # gate clarifies "how many?" instead.
    op = _add(LineItem(raw_text="salsa cups", sku="PCUP-2", supplier=S, quantity=None))
    out = apply_node({"cart_ops": [op], "draft_cart": Cart()})
    assert out["draft_cart"].is_empty()


def test_apply_node_does_not_land_a_line_that_exceeds_available_stock():
    # "200 cases of foil" when only 140 are in stock — don't commit the over-order;
    # the gate clarifies for a workable amount instead of banking 200.
    op = _add(LineItem(
        raw_text="foil", sku="FOIL-ROLL", supplier=S, quantity=200,
        quantity_on_hand=140, flags=[Flag.EXCEEDS_STOCK],
    ))
    out = apply_node({"cart_ops": [op], "draft_cart": Cart()})
    assert out["draft_cart"].is_empty()


def test_apply_node_does_not_land_an_out_of_stock_or_below_minimum_line():
    # Both flags mean the order is wrong as stated — the gate clarifies, the line
    # doesn't get banked. NEEDS_COMPANION (a valid parent) is covered separately.
    oos = _add(LineItem(raw_text="limes", sku="LIME-FRESH", supplier=S, quantity=2,
                        flags=[Flag.OUT_OF_STOCK]))
    below = _add(LineItem(raw_text="napkins", sku="NAPKIN-1", supplier=S, quantity=1,
                          flags=[Flag.BELOW_MINIMUM]))
    out = apply_node({"cart_ops": [oos, below], "draft_cart": Cart()})
    assert out["draft_cart"].is_empty()


def test_apply_node_still_lands_a_line_that_only_needs_a_companion():
    # NEEDS_COMPANION is a blocking flag but NOT a hold flag: the deli is a valid
    # order, so it lands while the desk upsells the lid.
    op = _add(LineItem(raw_text="deli", sku="DELI-16", product_name="16oz Deli Container",
                       supplier=S, quantity=3, flags=[Flag.NEEDS_COMPANION]))
    out = apply_node({"cart_ops": [op], "draft_cart": Cart()})
    assert [li.sku for li in out["draft_cart"].all_lines()] == ["DELI-16"]


def test_clarify_node_produces_questions_and_sets_status():
    op = _add(LineItem(raw_text="deli containers", confidence=0.3))
    out = clarify_node({"cart_ops": [op]})
    assert out["clarifications"]
    assert out["status"] == OrderStatus.NEEDS_CLARIFICATION


def test_draft_node_does_not_submit_a_running_draft_when_not_placing():
    cart = Cart(by_supplier={S: [LineItem(sku="DELI-16", supplier=S, quantity=2)]})
    out = draft_node({"draft_cart": cart}, FakeSupplier())
    assert out["status"] == OrderStatus.DRAFTED
    assert "confirmation" not in out  # nothing submitted — it's still a draft


def test_draft_node_submits_and_confirms_when_the_user_places_the_order():
    cart = Cart(by_supplier={S: [LineItem(sku="DELI-16", supplier=S, quantity=2)]})
    out = draft_node({"draft_cart": cart, "place_order": True}, FakeSupplier())
    assert out["status"] == OrderStatus.SUBMITTED
    assert out["confirmation"].order_id == "TEST-ORDER"


def test_build_clarifications_asks_about_a_low_confidence_item():
    questions = build_clarifications([LineItem(raw_text="deli containers", confidence=0.3)])
    assert len(questions) == 1


def test_build_clarifications_lists_the_candidate_options_when_present():
    line = LineItem(
        raw_text="deli containers",
        confidence=0.3,
        options=["8oz Deli Container", "16oz Deli Container", "32oz Deli Container"],
    )
    question = build_clarifications([line])[0]
    assert "8oz Deli Container" in question
    assert "16oz Deli Container" in question
    assert "32oz Deli Container" in question


def test_build_clarifications_names_the_companion_in_a_blocking_flag_question():
    line = LineItem(
        raw_text="deli", sku="DELI-16", product_name="16oz Deli Container",
        confidence=0.95, flags=[Flag.NEEDS_COMPANION],
        companions=[Companion(sku="LID-DELI", product_name="Deli Container Lid")],
    )
    questions = build_clarifications([line])
    assert len(questions) == 1
    assert "Deli Container Lid" in questions[0]


def test_build_clarifications_asks_how_many_for_a_missing_quantity():
    line = LineItem(
        raw_text="salsa cups", sku="PCUP-2", product_name="2oz Portion Cup",
        confidence=0.9, flags=[Flag.MISSING_QUANTITY],
    )
    question = build_clarifications([line])[0]
    assert "How many" in question
    assert "2oz Portion Cup" in question


def test_build_clarifications_names_the_requested_and_available_counts_when_over_stock():
    line = LineItem(
        raw_text="foil", sku="FOIL-ROLL", product_name="Aluminum Foil Roll",
        confidence=0.95, quantity=200, quantity_on_hand=140, flags=[Flag.EXCEEDS_STOCK],
    )
    question = build_clarifications([line])[0]
    assert "200" in question
    assert "140" in question


def test_build_clarifications_is_silent_for_clean_items():
    line = LineItem(raw_text="straws", sku="STRAW-WRAP", confidence=0.95, quantity=2)
    assert build_clarifications([line]) == []


# --- add_companions_node: an accepted offer becomes an ADD op, by SKU ----------
def _pending_lid_cart():
    parent = LineItem(
        sku="DELI-32", product_name="32oz Deli Container", supplier=S, quantity=3,
        companions=[Companion(sku="LID-DELI", product_name="Deli Container Lid")],
    )
    return Cart(by_supplier={S: [parent]})


def _deli_lid_repo(make_catalog_item):
    deli = make_catalog_item(
        sku="DELI-32", unit_size="32oz", case_pack=480, companion_skus=["LID-DELI"]
    )
    lid = make_catalog_item(
        sku="LID-DELI", product_name="Deli Container Lid", category="lids",
        unit_size="fits 8-32oz", case_pack=500,
    )
    return FakeCatalogRepo(items_by_sku={"DELI-32": deli, "LID-DELI": lid})


def _added(state, repo):
    return [op.item for op in add_companions_node(state, repo)["cart_ops"]]


def test_add_companions_adds_an_accepted_offer_by_sku_with_a_computed_quantity(make_catalog_item):
    state = {
        "cart_ops": [],
        "draft_cart": _pending_lid_cart(),
        "accepted_companions": [{"name": "Deli Container Lid", "quantity": None}],
    }
    added = _added(state, _deli_lid_repo(make_catalog_item))
    assert [li.sku for li in added] == ["LID-DELI"]
    assert added[0].quantity == 3  # ceil(3 * 480 / 500) = 3, covering 1440 containers


def test_add_companions_honors_a_user_stated_quantity(make_catalog_item):
    state = {
        "cart_ops": [],
        "draft_cart": _pending_lid_cart(),
        "accepted_companions": [{"name": "Deli Container Lid", "quantity": 2}],
    }
    added = _added(state, _deli_lid_repo(make_catalog_item))
    assert added[0].quantity == 2  # the explicit amount wins over the auto-calc


def test_add_companions_is_a_noop_when_nothing_is_accepted(make_catalog_item):
    state = {"cart_ops": [], "draft_cart": _pending_lid_cart(), "accepted_companions": []}
    assert _added(state, _deli_lid_repo(make_catalog_item)) == []


def test_add_companions_adds_only_the_accepted_subset(make_catalog_item):
    parent = LineItem(
        sku="DELI-32", product_name="32oz Deli Container", supplier=S, quantity=3,
        companions=[
            Companion(sku="LID-DELI", product_name="Deli Container Lid"),
            Companion(sku="LID-HOT", product_name="Hot Cup Lid"),
        ],
    )
    deli = make_catalog_item(sku="DELI-32", case_pack=480, companion_skus=["LID-DELI", "LID-HOT"])
    lid = make_catalog_item(
        sku="LID-DELI", product_name="Deli Container Lid", category="lids", case_pack=500
    )
    hot = make_catalog_item(
        sku="LID-HOT", product_name="Hot Cup Lid", category="lids", case_pack=1000
    )
    repo = FakeCatalogRepo(items_by_sku={"DELI-32": deli, "LID-DELI": lid, "LID-HOT": hot})
    state = {
        "cart_ops": [], "draft_cart": Cart(by_supplier={S: [parent]}),
        "accepted_companions": [{"name": "Deli Container Lid", "quantity": None}],
    }
    added = [li.sku for li in _added(state, repo)]
    assert added == ["LID-DELI"]  # the unaccepted Hot Cup Lid is left out


# --- companion coverage: which add-ons are still under-covered -----------------
def _coverage_repo(make_catalog_item):
    return FakeCatalogRepo(
        items_by_sku={
            "DELI-16": make_catalog_item(sku="DELI-16", case_pack=500, companion_skus=["LID-DELI"]),
            "DELI-32": make_catalog_item(sku="DELI-32", case_pack=480, companion_skus=["LID-DELI"]),
            "LID-DELI": make_catalog_item(
                sku="LID-DELI", product_name="Deli Container Lid", category="lids", case_pack=500
            ),
            "STRAW-WRAP": make_catalog_item(
                sku="STRAW-WRAP", product_name="Wrapped Straws", case_pack=10000
            ),
        }
    )


def test_offers_a_companion_when_its_line_has_no_lid_in_the_cart(make_catalog_item):
    cart = Cart(by_supplier={S: [LineItem(sku="DELI-16", supplier=S, quantity=3)]})
    pending = pending_companions(cart, _coverage_repo(make_catalog_item))
    assert [c.sku for c in pending] == ["LID-DELI"]


def test_does_not_offer_a_companion_the_cart_already_covers(make_catalog_item):
    # 3 cases of 16oz deli (1500 units) need 3 lid cases; 3 are present → covered.
    cart = Cart(by_supplier={S: [
        LineItem(sku="DELI-16", supplier=S, quantity=3),
        LineItem(sku="LID-DELI", supplier=S, quantity=3),
    ]})
    assert pending_companions(cart, _coverage_repo(make_catalog_item)) == []


def test_re_offers_when_a_present_lid_under_covers_an_added_size(make_catalog_item):
    # 3x 32oz (1440) + 2x 16oz (1000) = 2440 units need 5 lid cases; only 3 present.
    cart = Cart(by_supplier={S: [
        LineItem(sku="DELI-32", supplier=S, quantity=3),
        LineItem(sku="DELI-16", supplier=S, quantity=2),
        LineItem(sku="LID-DELI", supplier=S, quantity=3),
    ]})
    undercovered = _undercovered_companions(cart, _coverage_repo(make_catalog_item))
    assert [(c.sku, needed) for c, needed in undercovered] == [("LID-DELI", 5)]


def test_aggregates_a_shared_lid_across_two_deli_sizes(make_catalog_item):
    # No lid yet; the shared lid is offered once, sized to cover both deli lines.
    cart = Cart(by_supplier={S: [
        LineItem(sku="DELI-32", supplier=S, quantity=3),
        LineItem(sku="DELI-16", supplier=S, quantity=2),
    ]})
    undercovered = _undercovered_companions(cart, _coverage_repo(make_catalog_item))
    assert [(c.sku, needed) for c, needed in undercovered] == [("LID-DELI", 5)]


def test_no_companion_offer_for_a_line_without_companions(make_catalog_item):
    cart = Cart(by_supplier={S: [LineItem(sku="STRAW-WRAP", supplier=S, quantity=2)]})
    assert pending_companions(cart, _coverage_repo(make_catalog_item)) == []