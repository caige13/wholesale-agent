"""Ports — typing.Protocol seams for the external systems the agent depends on.

Two seams exist:

* ``CatalogRepository`` — static product data (names, aliases, case packs,
  minimums, lids) for semantic SKU resolution and rule checks. Backed by a
  vector store over ``kb/catalog.json``.
* ``SupplierGateway`` — dynamic supplier data (price, availability, orders).
  Backed by ``MockSupplierGateway`` over ``kb/supplier_inventory.json``.
* ``EscalationGateway`` — the human-handoff system (open a support ticket). Backed
  by ``MockEscalationGateway``.

The generic AI plumbing (chat model, embeddings, vector store) is seamed by
LangChain's own base classes, built by a factory in ``src.config``.
"""

from src.ports.catalog_repository import CatalogRepository
from src.ports.escalation_gateway import EscalationGateway
from src.ports.supplier_gateway import SupplierGateway

__all__ = ["CatalogRepository", "EscalationGateway", "SupplierGateway"]