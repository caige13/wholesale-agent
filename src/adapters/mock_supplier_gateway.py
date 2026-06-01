"""JSON-backed mock supplier gateway — stands in for the real supplier API.

Implements ``SupplierGateway`` over ``kb/supplier_inventory.json`` (per-SKU stock,
lead time, and price). Keyless and deterministic, so it serves the runtime and is
unit-tested directly — no network, no API key. Price/stock live here, never in the
RAG catalog (``kb/catalog.json``), keeping the static corpus and dynamic data
cleanly split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.domain.models import InventoryStatus, LineItem, OrderConfirmation

# Repo-root-relative default: src/adapters/ -> repo root -> kb/supplier_inventory.json
_DEFAULT_INVENTORY_PATH = Path(__file__).resolve().parents[2] / "kb" / "supplier_inventory.json"


class MockSupplierGateway:
    """Serves price/stock from ``kb/supplier_inventory.json`` and fakes order submission.

    An unknown SKU is treated as in stock (``in_stock=True``) with no price — a
    non-blocking default, so a catalog item missing from the inventory file never
    silently halts an order.
    """

    def __init__(
        self,
        inventory_path: str | Path = _DEFAULT_INVENTORY_PATH,
        supplier: str = "acme-foodservice",
    ) -> None:
        self.supplier = supplier
        rows = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
        self._by_sku: dict[str, dict] = {}
        for row in rows:
            sku = row["sku"]
            if sku in self._by_sku:
                raise ValueError(f"duplicate SKU in supplier inventory: {sku}")
            self._by_sku[sku] = row

    def get_price(self, sku: str) -> float | None:
        row = self._by_sku.get(sku)
        return row.get("price") if row else None

    def check_inventory(self, sku: str) -> InventoryStatus:
        row = self._by_sku.get(sku)
        if row is None:
            return InventoryStatus(in_stock=True)
        return InventoryStatus(
            in_stock=row["in_stock"],
            quantity_on_hand=row.get("quantity_on_hand", 0),
            lead_time_days=row.get("lead_time_days", 0),
        )

    def submit_order(self, items: list[LineItem]) -> OrderConfirmation:
        total = sum(
            line.quantity * line.unit_price
            for line in items
            if line.quantity is not None and line.unit_price is not None
        )
        return OrderConfirmation(
            order_id=self._order_id(items),
            supplier=self.supplier,
            total=round(total, 2) if total else None,
        )

    def _order_id(self, items: list[LineItem]) -> str:
        """A stable id from the ordered (sku, quantity) pairs — deterministic so
        the same cart always confirms the same id (no clock / randomness)."""
        payload = ";".join(sorted(f"{line.sku}:{line.quantity}" for line in items))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8].upper()
        return f"{self.supplier.upper()}-{digest}"