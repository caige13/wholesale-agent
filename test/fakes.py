"""Shared test doubles — keyless fakes for the ports the graph/UX depend on.

These live in a module (not ``conftest.py``) because tests instantiate them as
classes, including inside module-level helpers where fixtures aren't available.
``conftest.py`` stays for the data-object *factory fixtures* (make_catalog_item,
make_candidate). Importable via ``from test.fakes import ...`` (pythonpath = ".").
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from src.app.graph.llm_nodes import IntentResult, ParsedOrder
from src.domain.cart import Cart
from src.domain.models import CatalogItem, InventoryStatus, OrderConfirmation
from src.ports.order_agent import AgentResult

SUPPLIER = "acme-foodservice"


class ScriptedModel:
    """Fake chat model for the order graph — keyless, no real LLM.

    Covers the three ways the graph calls a model:
    - ``with_structured_output`` for intent classification + order parsing (canned).
    - ``bind_tools`` for the QA subgraph's assistant<->tools loop. With ``tool_steps``
      left ``None`` the bound model answers immediately (no tool calls, content =
      ``answer``); pass ``tool_steps`` to script a sequence — a list of
      ``{"name", "args"}`` dicts becomes an ``AIMessage`` with those ``tool_calls``,
      a plain string becomes the final answer ``AIMessage``.
    """

    def __init__(self, intent, parsed=None, answer="", tool_steps=None):
        self._intent = intent
        self._parsed = parsed or ParsedOrder()
        self._answer = answer
        self._tool_steps = tool_steps

    def with_structured_output(self, schema):
        result = IntentResult(intent=self._intent) if schema is IntentResult else self._parsed
        return SimpleNamespace(invoke=lambda _prompt: result)

    def bind_tools(self, _tools):
        steps = list(self._tool_steps) if self._tool_steps is not None else None

        def _invoke(_messages):
            if steps is None:  # no script: the model just answers
                return AIMessage(content=self._answer)
            step = steps.pop(0)
            if isinstance(step, str):  # a string step is the final answer
                return AIMessage(content=step)
            tool_calls = [  # a list step is one round of tool calls
                {
                    "name": c["name"],
                    "args": c.get("args", {}),
                    "id": c.get("id", f"call_{i}"),
                    "type": "tool_call",
                }
                for i, c in enumerate(step)
            ]
            return AIMessage(content="", tool_calls=tool_calls)

        return SimpleNamespace(invoke=_invoke)


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