"""Shared test doubles — keyless fakes for the ports the graph/UX depend on.

These live in a module (not ``conftest.py``) because tests instantiate them as
classes, including inside module-level helpers where fixtures aren't available.
``conftest.py`` stays for the data-object *factory fixtures* (make_catalog_item,
make_candidate). Importable via ``from test.fakes import ...`` (pythonpath = ".").
"""

from __future__ import annotations

from types import SimpleNamespace

from src.app.graph.llm_nodes import IntentResult, ParsedOrder
from src.domain.cart import Cart
from src.domain.models import CatalogItem, InventoryStatus, OrderConfirmation
from src.ports.order_agent import AgentResult

SUPPLIER = "acme-foodservice"


class ScriptedModel:
    """Fake chat model: canned intent + parse results, and a canned QA answer."""

    def __init__(self, intent, parsed=None, answer=""):
        self._intent = intent
        self._parsed = parsed or ParsedOrder()
        self._answer = answer

    def with_structured_output(self, schema):
        result = IntentResult(intent=self._intent) if schema is IntentResult else self._parsed
        return SimpleNamespace(invoke=lambda _prompt: result)

    def invoke(self, _prompt):
        return SimpleNamespace(content=self._answer)


class FakeCatalog:
    """In-memory CatalogRepository: ``get``/``all`` from items, ``find_candidates``
    from a phrase→candidates map (empty list when a phrase isn't mapped)."""

    def __init__(self, items_by_sku=None, candidates_by_phrase=None):
        self._items = items_by_sku or {}
        self._candidates = candidates_by_phrase or {}

    def get(self, sku):
        return self._items.get(sku)

    def all(self):
        return list(self._items.values())

    def find_candidates(self, query, k=5):
        return self._candidates.get(query.strip().lower(), [])


class FakeSupplier:
    """Fake supplier gateway: every SKU in stock and unpriced unless named in
    ``out_of_stock`` / ``prices``."""

    supplier = SUPPLIER

    def __init__(self, out_of_stock=None, prices=None):
        self._out_of_stock = set(out_of_stock or ())
        self._prices = prices or {}

    def get_price(self, sku):
        return self._prices.get(sku)

    def check_inventory(self, sku):
        return InventoryStatus(in_stock=sku not in self._out_of_stock)

    def submit_order(self, items):
        return OrderConfirmation(order_id="TEST-ORDER", supplier=self.supplier)


class FakeOrderAgent:
    """Stub inner agent: returns a preset AgentResult and records each call as
    ``(message, cart, history)``; ``last_history`` reads the most recent one."""

    def __init__(self, result: AgentResult):
        self._result = result
        self.calls: list[tuple[str, Cart, list | None]] = []

    @property
    def last_history(self):
        return self.calls[-1][2] if self.calls else None

    def run(self, message: str, cart: Cart, history=None, *, trace=None) -> AgentResult:
        # trace is observability-only; the stub ignores it (records just the inputs).
        self.calls.append((message, cart, history))
        return self._result


def catalog_item(sku, name, **kw) -> CatalogItem:
    """A minimal CatalogItem for graph tests (category/unit_size are placeholders)."""
    return CatalogItem(
        sku=sku, product_name=name, category="x", unit_size="x", case_pack=100,
        supplier=SUPPLIER, **kw,
    )