"""validate_rules — static-catalog business rules (spec §4).

Pure: returns a new LineItem with quantity rounded to whole cases and the
catalog-derived flags set (case-pack rounding, below-minimum, needs-lids).
Out-of-stock is NOT here — stock is dynamic supplier-API data set by the
inventory check.
"""

from src.domain.models import Flag, LineItem
from src.domain.rules import validate_rules


def test_rounds_units_up_to_whole_cases_when_not_an_exact_multiple(make_catalog_item):
    item = LineItem(sku="DELI-16", unit_quantity=1200)
    out = validate_rules(item, make_catalog_item(case_pack=500))
    assert out.quantity == 3
    assert Flag.ROUNDED_TO_CASE_PACK in out.flags


def test_does_not_flag_rounding_when_units_are_an_exact_multiple(make_catalog_item):
    item = LineItem(sku="DELI-16", unit_quantity=1000)
    out = validate_rules(item, make_catalog_item(case_pack=500))
    assert out.quantity == 2
    assert Flag.ROUNDED_TO_CASE_PACK not in out.flags


def test_flags_below_minimum_when_quantity_is_under_min_order(make_catalog_item):
    out = validate_rules(LineItem(sku="X", quantity=1), make_catalog_item(min_order=2))
    assert Flag.BELOW_MINIMUM in out.flags


def test_does_not_flag_below_minimum_when_quantity_meets_min_order(make_catalog_item):
    out = validate_rules(LineItem(sku="X", quantity=2), make_catalog_item(min_order=2))
    assert Flag.BELOW_MINIMUM not in out.flags


def test_flags_needs_companion_when_item_has_companions(make_catalog_item):
    item = make_catalog_item(companion_skus=["LID-DELI"])
    out = validate_rules(LineItem(sku="DELI-16", quantity=3), item)
    assert Flag.NEEDS_COMPANION in out.flags


def test_does_not_flag_needs_companion_when_item_has_no_companions(make_catalog_item):
    out = validate_rules(LineItem(sku="X", quantity=3), make_catalog_item(companion_skus=None))
    assert Flag.NEEDS_COMPANION not in out.flags


def test_never_sets_out_of_stock_flag(make_catalog_item):
    # Stock is dynamic supplier data — validate_rules must never decide it.
    out = validate_rules(LineItem(sku="X", quantity=3), make_catalog_item())
    assert Flag.OUT_OF_STOCK not in out.flags


def test_returns_a_new_item_without_mutating_the_input(make_catalog_item):
    item = LineItem(sku="X", quantity=1)
    out = validate_rules(item, make_catalog_item(min_order=2, companion_skus=["LID-DELI"]))
    assert out is not item
    assert item.flags == []
    assert item.quantity == 1


def test_preserves_pre_existing_flags(make_catalog_item):
    item = LineItem(sku="X", quantity=1, flags=[Flag.AMBIGUOUS_SIZE])
    out = validate_rules(item, make_catalog_item(min_order=2))
    assert Flag.AMBIGUOUS_SIZE in out.flags
    assert Flag.BELOW_MINIMUM in out.flags


def test_does_not_duplicate_a_flag_that_already_exists(make_catalog_item):
    item = LineItem(sku="DELI-16", quantity=3, flags=[Flag.NEEDS_COMPANION])
    out = validate_rules(item, make_catalog_item(companion_skus=["LID-DELI"]))
    assert out.flags.count(Flag.NEEDS_COMPANION) == 1