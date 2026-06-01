"""Supplier gateway port — the supplier-API seam (price / stock / orders).

The agent's window into *dynamic* supplier data, the counterpart to
``CatalogRepository``'s static catalog: unit price, live availability, and order
submission. Adapters translate the supplier's wire format behind this boundary so
``check_inventory_node`` / ``draft_node`` speak only the domain language
(``InventoryStatus`` / ``OrderConfirmation``).

Price and stock deliberately do NOT live in the RAG catalog — they change, so
they belong behind this gateway, not in ``kb/catalog.json``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.domain.models import InventoryStatus, LineItem, OrderConfirmation


@runtime_checkable
class SupplierGateway(Protocol):
    supplier: str

    def get_price(self, sku: str) -> float | None:
        """Return the unit price for this SKU, or ``None`` if it isn't priced."""
        ...

    def check_inventory(self, sku: str) -> InventoryStatus:
        """Return live stock for this SKU (``in_stock`` drives ``OUT_OF_STOCK``)."""
        ...

    def submit_order(self, items: list[LineItem]) -> OrderConfirmation:
        """Place the order with the supplier and return its confirmation."""
        ...