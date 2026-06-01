r"""MockSupplierGateway — serves price/stock from kb/supplier_inventory.json."""

import pytest

from src.adapters import MockSupplierGateway
from src.domain.models import LineItem
from src.ports import SupplierGateway


@pytest.fixture
def supplier() -> MockSupplierGateway:
    return MockSupplierGateway()


def test_adapter_satisfies_the_supplier_gateway_port(supplier: MockSupplierGateway):
    # Guards the hexagonal seam: the JSON adapter must honor the port contract.
    assert isinstance(supplier, SupplierGateway)


def test_get_price_returns_the_price_for_a_known_sku(supplier: MockSupplierGateway):
    assert supplier.get_price("DELI-16") == 41.75


def test_get_price_returns_none_for_an_unknown_sku(supplier: MockSupplierGateway):
    assert supplier.get_price("NOPE-999") is None


def test_check_inventory_reports_a_stocked_sku_as_in_stock(supplier: MockSupplierGateway):
    status = supplier.check_inventory("DELI-16")
    assert status.in_stock
    assert status.quantity_on_hand > 0


def test_check_inventory_reports_limes_out_of_stock_with_a_lead_time(
    supplier: MockSupplierGateway,
):
    status = supplier.check_inventory("LIME-FRESH")
    assert not status.in_stock
    assert status.lead_time_days > 0


def test_check_inventory_treats_an_unknown_sku_as_in_stock(supplier: MockSupplierGateway):
    # Non-blocking default: a catalog item missing from the inventory file must
    # not silently halt an order.
    assert supplier.check_inventory("NOPE-999").in_stock


def test_submit_order_returns_a_confirmation_with_supplier_and_computed_total(
    supplier: MockSupplierGateway,
):
    items = [
        LineItem(raw_text="deli", sku="DELI-16", quantity=2, unit_price=41.75),
        LineItem(raw_text="forks", sku="FORK-PLAS", quantity=1, unit_price=19.95),
    ]
    confirmation = supplier.submit_order(items)
    assert confirmation.supplier == "acme-foodservice"
    assert confirmation.order_id
    assert confirmation.total == pytest.approx(2 * 41.75 + 19.95)


def test_submit_order_is_deterministic_for_the_same_lines(supplier: MockSupplierGateway):
    items = [LineItem(raw_text="deli", sku="DELI-16", quantity=2, unit_price=41.75)]
    assert supplier.submit_order(items).order_id == supplier.submit_order(items).order_id