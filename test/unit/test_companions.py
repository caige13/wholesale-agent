"""companion_case_count + pending_companions — the deterministic add-on helpers.

The LLM does no companion math or bookkeeping: these pure functions size the
add-on order (cover the parent without going under) and derive which offers are
still open from the cart itself, so a "yes" needs no extra cross-turn state.
"""

from src.domain.cart import Cart
from src.domain.companions import companion_case_count, pending_companions
from src.domain.models import Companion, LineItem

S = "acme-foodservice"


def test_sizes_a_companion_to_cover_the_parent_without_going_under():
    # 3 cases of 32oz deli = 1440 containers; lids packed 500/case → 3 (1500 ≥ 1440).
    assert companion_case_count(1440, 500) == 3


def test_rounds_up_to_the_next_whole_case_rather_than_under():
    assert companion_case_count(1001, 500) == 3  # 2 cases (1000) would be short


def test_returns_at_least_one_case_for_an_exact_or_unknown_parent():
    assert companion_case_count(1000, 500) == 2
    assert companion_case_count(0, 500) == 1  # unknown parent count → still offer a case


def test_lists_offers_for_lines_whose_companion_is_not_in_the_cart():
    deli = LineItem(
        sku="DELI-32", supplier=S, quantity=3,
        companions=[Companion(sku="LID-DELI", product_name="Deli Container Lid")],
    )
    pending = pending_companions(Cart(by_supplier={S: [deli]}))
    assert [c.sku for c in pending] == ["LID-DELI"]


def test_drops_an_offer_once_its_companion_is_in_the_cart():
    deli = LineItem(
        sku="DELI-32", supplier=S, quantity=3,
        companions=[Companion(sku="LID-DELI", product_name="Deli Container Lid")],
    )
    lid = LineItem(sku="LID-DELI", supplier=S, quantity=3)
    assert pending_companions(Cart(by_supplier={S: [deli, lid]})) == []


def test_dedups_a_companion_shared_by_two_lines():
    a = LineItem(
        sku="DELI-16", supplier=S, quantity=1,
        companions=[Companion(sku="LID-DELI", product_name="Deli Container Lid")],
    )
    b = LineItem(
        sku="DELI-32", supplier=S, quantity=1,
        companions=[Companion(sku="LID-DELI", product_name="Deli Container Lid")],
    )
    assert [c.sku for c in pending_companions(Cart(by_supplier={S: [a, b]}))] == ["LID-DELI"]


def test_no_offers_for_a_line_without_companions():
    line = LineItem(sku="STRAW-WRAP", supplier=S, quantity=1)
    assert pending_companions(Cart(by_supplier={S: [line]})) == []