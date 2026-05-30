"""Deterministic graph nodes — thin wrappers over the pure functions.

The per-turn carrier is ``cart_ops``: each node maps OrderState in → a partial
update out, enriching/applying the ops. Keyless (a fake catalog repo stands in
for the FAISS retriever); the LLM nodes are contract-tested separately.
"""

from src.app.graph.nodes import (
    apply_node,
    build_clarifications,
    clarify_node,
    draft_node,
    redact_node,
    resolve_node,
    validate_node,
)
from src.domain.cart import Cart
from src.domain.models import CartOp, CartOpKind, Flag, LineItem, OrderStatus, ResolutionCandidate

S = "acme-foodservice"


class FakeCatalogRepo:
    """In-memory CatalogRepository stand-in for keyless node tests."""

    def __init__(self, items_by_sku=None, candidates_by_phrase=None):
        self._items = items_by_sku or {}
        self._candidates = candidates_by_phrase or {}

    def get(self, sku):
        return self._items.get(sku)

    def all(self):
        return list(self._items.values())

    def find_candidates(self, query, k=5):
        return self._candidates.get(query.strip().lower(), [])


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


def test_validate_node_raises_flags_on_the_op_item(make_catalog_item):
    deli = make_catalog_item(sku="DELI-16", requires_lids=True)
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli})
    state = {"cart_ops": [_add(LineItem(raw_text="deli", sku="DELI-16", quantity=3))]}
    out = validate_node(state, repo)
    assert Flag.NEEDS_LIDS in out["cart_ops"][0].item.flags


def test_validate_node_skips_remove_ops(make_catalog_item):
    deli = make_catalog_item(sku="DELI-16", requires_lids=True)
    repo = FakeCatalogRepo(items_by_sku={"DELI-16": deli})
    remove = CartOp(op=CartOpKind.REMOVE, item=LineItem(raw_text="deli", sku="DELI-16"))
    out = validate_node({"cart_ops": [remove]}, repo)
    assert out["cart_ops"][0].item.flags == []  # a removal isn't validated for lids etc.


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


def test_draft_node_sets_status_to_drafted():
    assert draft_node({})["status"] == OrderStatus.DRAFTED


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


def test_build_clarifications_asks_about_a_blocking_flag():
    line = LineItem(
        raw_text="deli", sku="DELI-16", product_name="16oz Deli Container",
        confidence=0.95, flags=[Flag.NEEDS_LIDS],
    )
    assert len(build_clarifications([line])) == 1


def test_build_clarifications_is_silent_for_clean_items():
    line = LineItem(raw_text="straws", sku="STRAW-WRAP", confidence=0.95)
    assert build_clarifications([line]) == []