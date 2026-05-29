"""Ports — typing.Protocol seams for the external systems the agent depends on.

So far only the RAG knowledge base seam exists:

* ``CatalogRepository`` — static product data (names, aliases, case packs,
  minimums, lids) for semantic SKU resolution and rule checks. Backed by a
  vector store over ``kb/catalog.json``.

The supplier-API seam (``SupplierGateway`` — price/availability/orders) is
designed but deferred; it returns test-first when the supplier tools land. The
generic AI plumbing (chat model, embeddings, vector store) is seamed by
LangChain's own base classes, built by a factory in ``src.config``.
"""

from src.ports.catalog_repository import CatalogRepository

__all__ = ["CatalogRepository"]