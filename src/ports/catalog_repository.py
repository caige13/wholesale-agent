"""Catalog repository port — the RAG knowledge base seam.

This is the agent's window into the *static* product catalog: names, aliases,
case packs, minimums, lids. It speaks the domain language (``CatalogItem`` /
``ResolutionCandidate``), so adapters translate JSON rows or LangChain
retriever ``Document``s behind this boundary and ``resolve_skus`` /
``validate_rules`` never see a raw ``Document``.

It is NOT the supplier API: price, availability, and stock are dynamic and live
behind ``SupplierGateway``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.domain.models import CatalogItem, ResolutionCandidate


@runtime_checkable
class CatalogRepository(Protocol):
    def get(self, sku: str) -> CatalogItem | None:
        """Return the item with this SKU, or ``None`` if unknown."""
        ...

    def all(self) -> list[CatalogItem]:
        """Return every catalog item (used to seed the vector store and rules)."""
        ...

    def find_candidates(
        self, query: str, k: int = 5, suppliers: list[str] | None = None
    ) -> list[ResolutionCandidate]:
        """Return up to ``k`` candidate SKUs for a free-text query, best first.

        The RAG retrieval seam: the FAISS-backed adapter embeds the query and
        returns nearest catalog rows as scored ``ResolutionCandidate``s for
        ``resolve_skus`` to disambiguate.

        ``suppliers`` scopes the search to the customer's selected tenants — only
        those suppliers' catalogs are searched. ``None`` or an empty list means
        search across all suppliers (the single-tenant default). This is the
        multi-tenant seam: the same signature holds whether the adapter is FAISS
        (metadata filter) or a future vector DB (a ``WHERE supplier IN (…)``
        pre-filter); see ``docs/multi-tenant-suppliers.md``.
        """
        ...