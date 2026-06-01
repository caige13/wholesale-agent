r"""check_inventory_node — enriches cart ops with live supplier price/stock."""

from src.app.graph.nodes import check_inventory_node
from src.domain.models import CartOp, CartOpKind, Flag, LineItem
from test.fakes import FakeSupplier


def _add(sku, **kw):
    return CartOp(op=CartOpKind.ADD, item=LineItem(raw_text=sku, sku=sku, **kw))


def _items(result):
    return [op.item for op in result["cart_ops"]]


def test_sets_out_of_stock_and_does_not_touch_in_stock_lines():
    state = {"cart_ops": [_add("LIME-FRESH"), _add("DELI-16")]}
    supplier = FakeSupplier(out_of_stock={"LIME-FRESH"})
    lime, deli = _items(check_inventory_node(state, supplier))
    assert Flag.OUT_OF_STOCK in lime.flags
    assert Flag.OUT_OF_STOCK not in deli.flags


def test_fills_unit_price_from_the_gateway():
    state = {"cart_ops": [_add("DELI-16")]}
    supplier = FakeSupplier(prices={"DELI-16": 41.75})
    (deli,) = _items(check_inventory_node(state, supplier))
    assert deli.unit_price == 41.75


def test_skips_remove_ops_and_unresolved_lines():
    remove = CartOp(op=CartOpKind.REMOVE, item=LineItem(raw_text="limes", sku="LIME-FRESH"))
    unresolved = CartOp(op=CartOpKind.ADD, item=LineItem(raw_text="mystery item"))
    state = {"cart_ops": [remove, unresolved]}
    supplier = FakeSupplier(out_of_stock={"LIME-FRESH"})
    removed, mystery = _items(check_inventory_node(state, supplier))
    # A removal isn't checked even though the SKU is out of stock.
    assert Flag.OUT_OF_STOCK not in removed.flags
    assert mystery.flags == []


def test_preserves_flags_already_on_the_line():
    state = {"cart_ops": [_add("DELI-16", flags=[Flag.NEEDS_COMPANION])]}
    supplier = FakeSupplier(out_of_stock={"DELI-16"})
    (deli,) = _items(check_inventory_node(state, supplier))
    assert Flag.NEEDS_COMPANION in deli.flags
    assert Flag.OUT_OF_STOCK in deli.flags