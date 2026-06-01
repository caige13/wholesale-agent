"""Deterministic graph nodes — thin wrappers over the pure functions.

The per-turn carrier is ``cart_ops``: each node maps OrderState in → a partial
update out, enriching/applying the ops. Keyless (a fake catalog repo stands in
for the FAISS retriever); the LLM nodes are contract-tested separately.
"""

from src.app.graph.nodes import (
    add_companions_node,
    apply_node,
    build_clarifications,
    clarify_node,
    draft_node,
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


def test_validate_node_raises_companion_flag_and_names_the_offer(make_catalog_item):
    deli = make_catalog_item(sku="DELI-16", companion_skus=["LID-DELI"])
    lid = make_catalog_item(sku="LID-DELI", product_name="Deli Container Lid", category="lids")
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli, "LID-DELI": lid})
    state = {"cart_ops": [_add(LineItem(raw_text="deli", sku="DELI-16", quantity=3))]}
    item = validate_node(state, repo)["cart_ops"][0].item
    assert Flag.NEEDS_COMPANION in item.flags
    assert [c.product_name for c in item.companions] == ["Deli Container Lid"]


def test_validate_node_makes_no_offer_when_the_companion_is_already_in_the_cart(make_catalog_item):
    deli = make_catalog_item(sku="DELI-16", companion_skus=["LID-DELI"])
    lid = make_catalog_item(sku="LID-DELI", product_name="Deli Container Lid", category="lids")
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli, "LID-DELI": lid})
    cart = Cart(by_supplier={S: [LineItem(sku="LID-DELI", supplier=S, quantity=1)]})
    state = {
        "cart_ops": [_add(LineItem(raw_text="deli", sku="DELI-16", quantity=3))],
        "draft_cart": cart,
    }
    item = validate_node(state, repo)["cart_ops"][0].item
    assert Flag.NEEDS_COMPANION not in item.flags  # lids already present → no nudge


def test_validate_node_does_not_re_ask_for_a_companion_on_a_quantity_change(make_catalog_item):
    # Bumping the quantity of a container already in the cart shouldn't re-nag.
    deli = make_catalog_item(sku="DELI-16", companion_skus=["LID-DELI"])
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli})
    line = LineItem(raw_text="deli", sku="DELI-16", quantity=3)
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


def test_clarify_node_produces_questions_and_sets_status():
    op = _add(LineItem(raw_text="deli containers", confidence=0.3))
    out = clarify_node({"cart_ops": [op]})
    assert out["clarifications"]
    assert out["status"] == OrderStatus.NEEDS_CLARIFICATION


def test_draft_node_sets_status_to_drafted_and_confirms_the_order():
    out = draft_node({}, FakeSupplier())
    assert out["status"] == OrderStatus.DRAFTED
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


def test_build_clarifications_is_silent_for_clean_items():
    line = LineItem(raw_text="straws", sku="STRAW-WRAP", confidence=0.95)
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