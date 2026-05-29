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


def test_appends_a_new_line_when_adding_an_unseen_sku():
    out = Cart().apply([op(CartOpKind.ADD, line("DELI-16", 2))])
    assert out == Cart(by_supplier={S: [line("DELI-16", 2)]})


def test_merges_quantity_when_adding_an_existing_sku():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.ADD, line("DELI-16", 3))])
    assert len(out.by_supplier[S]) == 1
    assert out.by_supplier[S][0].quantity == 5


def test_appends_to_an_existing_supplier_when_adding_a_different_sku():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.ADD, line("LIME-FRESH", 1, "Fresh Limes"))])
    assert {li.sku for li in out.by_supplier[S]} == {"DELI-16", "LIME-FRESH"}


def test_set_quantity_replaces_the_line_rather_than_appending():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.SET_QUANTITY, line("DELI-16", 3))])
    assert len(out.by_supplier[S]) == 1
    assert out.by_supplier[S][0].quantity == 3


def test_set_quantity_appends_when_the_sku_is_absent():
    out = Cart().apply([op(CartOpKind.SET_QUANTITY, line("DELI-16", 3))])
    assert out.by_supplier[S][0].quantity == 3


def test_remove_drops_the_targeted_line():
    cart = Cart(by_supplier={S: [line("DELI-16", 2), line("LIME-FRESH", 1, "Fresh Limes")]})
    out = cart.apply([op(CartOpKind.REMOVE, line("LIME-FRESH", 1, "Fresh Limes"))])
    assert {li.sku for li in out.by_supplier[S]} == {"DELI-16"}


def test_drops_the_supplier_key_when_its_last_line_is_removed():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.REMOVE, line("DELI-16", 2))])
    assert S not in out.by_supplier


def test_apply_returns_a_new_cart_without_mutating_the_receiver():
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    cart.apply([op(CartOpKind.ADD, line("DELI-16", 3))])
    assert cart.by_supplier[S][0].quantity == 2


def test_applies_multiple_ops_in_order():
    out = Cart().apply(
        [
            op(CartOpKind.ADD, line("DELI-16", 2)),
            op(CartOpKind.ADD, line("DELI-16", 1)),
            op(CartOpKind.SET_QUANTITY, line("DELI-16", 10)),
        ]
    )
    assert out.by_supplier[S][0].quantity == 10


def test_is_empty_reflects_whether_the_cart_has_lines():
    assert Cart().is_empty()
    assert not Cart(by_supplier={S: [line("DELI-16", 1)]}).is_empty()


def test_all_lines_flattens_lines_across_suppliers():
    cart = Cart(by_supplier={S: [line("DELI-16", 2), line("LIME-FRESH", 1, "Fresh Limes")]})
    assert len(cart.all_lines()) == 2


def test_set_quantity_with_no_quantity_leaves_the_existing_line_unchanged():
    # A set_quantity that carries no quantity must NOT silently null the line.
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.SET_QUANTITY, LineItem(sku="DELI-16", supplier=S))])
    assert out.by_supplier[S][0].quantity == 2


def test_applying_no_ops_returns_an_equal_cart():
    # The question-turn guardrail: zero cart_ops must leave the cart unchanged.
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    assert cart.apply([]) == cart


def test_removing_an_absent_sku_is_a_noop_without_a_phantom_supplier():
    # setdefault() transiently creates the supplier bucket; the apply() filter is
    # the only thing that keeps a missed remove from leaving an empty group.
    cart = Cart(by_supplier={S: [line("DELI-16", 2)]})
    out = cart.apply([op(CartOpKind.REMOVE, line("MISSING-1", 1, supplier="other-co"))])
    assert out == cart
    assert "other-co" not in out.by_supplier