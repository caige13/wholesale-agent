"""Cart.apply — pure cart mutation owned by the Cart aggregate (spec §4).

The single place the cart changes: the agent emits cart_ops, Cart.apply applies
them and returns a NEW cart. add merges an existing SKU / appends a new one;
set_quantity replaces (never duplicates); remove drops the line; an emptied
supplier disappears. Pure — never mutates the receiver.
"""

from src.domain.cart import Cart
from src.domain.models import CartOp, CartOpKind, LineItem

S = "acme-foodservice"


def line(sku: str, qty: int, name: str = "Item", supplier: str = S) -> LineItem:
    return LineItem(sku=sku, product_name=name, supplier=supplier, quantity=qty)


def op(kind: CartOpKind, item: LineItem) -> CartOp:
    return CartOp(op=kind, item=item)


def test_add_new_sku_appends_line():
    out = Cart().apply([op(CartOpKind.ADD, line("DELI-16", 2))])
    assert out == Cart(by_supplier={S: [line("DELI-16", 2)]})


def test_add_existing_sku_merges_quantity():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.ADD, line("DELI-16", 3))])
    assert len(out.by_supplier[S]) == 1
    assert out.by_supplier[S][0].quantity == 5


def test_add_new_sku_to_existing_supplier_appends():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.ADD, line("LIME-FRESH", 1, "Fresh Limes"))])
    assert {li.sku for li in out.by_supplier[S]} == {"DELI-16", "LIME-FRESH"}


def test_set_quantity_replaces_does_not_append():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.SET_QUANTITY, line("DELI-16", 3))])
    assert len(out.by_supplier[S]) == 1
    assert out.by_supplier[S][0].quantity == 3


def test_set_quantity_on_absent_sku_appends():
    out = Cart().apply([op(CartOpKind.SET_QUANTITY, line("DELI-16", 3))])
    assert out.by_supplier[S][0].quantity == 3


def test_remove_drops_the_line():
    cart = Cart(by_supplier={S: [line("DELI-16", 2), line("LIME-FRESH", 1, "Fresh Limes")]})
    out = cart.apply([op(CartOpKind.REMOVE, line("LIME-FRESH", 1, "Fresh Limes"))])
    assert {li.sku for li in out.by_supplier[S]} == {"DELI-16"}


def test_removing_last_line_drops_the_supplier_key():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.REMOVE, line("DELI-16", 2))])
    assert S not in out.by_supplier


def test_apply_does_not_mutate_the_receiver():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    cart.apply([op(CartOpKind.ADD, line("DELI-16", 3))])
    assert cart.by_supplier[S][0].quantity == 2


def test_multiple_ops_apply_in_order():
    out = Cart().apply(
        [
            op(CartOpKind.ADD, line("DELI-16", 2)),
            op(CartOpKind.ADD, line("DELI-16", 1)),
            op(CartOpKind.SET_QUANTITY, line("DELI-16", 10)),
        ]
    )
    assert out.by_supplier[S][0].quantity == 10


def test_empty_cart_reports_empty():
    assert Cart().is_empty()
    assert not Cart(by_supplier={S: [line("DELI-16", 1)]}).is_empty()


def test_all_lines_flattens_suppliers():
    cart = Cart(by_supplier={S: [line("DELI-16", 2), line("LIME-FRESH", 1, "Fresh Limes")]})
    assert len(cart.all_lines()) == 2