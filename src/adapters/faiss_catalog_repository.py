"""FAISS-backed catalog repository — the RAG retrieval adapter.

Implements the full ``CatalogRepository`` port: ``get``/``all`` over the loaded
items, and ``find_candidates`` as semantic search. It embeds
``product_name + aliases + category`` per SKU into a FAISS index and translates
LangChain ``Document`` hits back into domain ``ResolutionCandidate``s — the
anti-corruption boundary, so ``resolve_skus`` never sees a ``Document``.

Multi-tenant (Phase 0, see ``docs/multi-tenant-suppliers.md``): every vector is
tagged with its ``supplier`` in the Document metadata, so ``find_candidates`` can
scope a search to the customer's selected suppliers via a metadata filter. The
filter is post-retrieval in FAISS, so we over-fetch (``fetch_k`` = corpus size) to
keep the scoped top-k exact while the catalog is small.

Kept out of ``src.adapters.__init__`` on purpose: importing it pulls FAISS +
embeddings, so the keyless unit suite never loads it.
"""

from __future__ import annotations

import warnings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.domain.models import CatalogItem, ResolutionCandidate


def _embedding_text(item: CatalogItem) -> str:
    """What we embed for a SKU — name + aliases + category make "salsa cups" hit."""
    return " ".join([item.product_name, *item.aliases, item.category])


def _clamp01(score: float) -> float:
    return max(0.0, min(1.0, score))


class FaissCatalogRepository:
    """CatalogRepository backed by an in-memory FAISS index over the catalog."""

    def __init__(self, items: list[CatalogItem], embeddings: Embeddings):
        self._by_sku: dict[str, CatalogItem] = {item.sku: item for item in items}
        # Composite (supplier, sku) key for mapping a hit back to its item: SKUs can
        # repeat across suppliers in a multi-tenant catalog, so the metadata carries
        # both and the lookup uses the pair.
        self._by_key: dict[tuple[str, str], CatalogItem] = {
            (item.supplier, item.sku): item for item in items
        }
        documents = [
            Document(
                page_content=_embedding_text(item),
                metadata={"sku": item.sku, "supplier": item.supplier},
            )
            for item in items
        ]
        self._store = FAISS.from_documents(documents, embeddings)

    def get(self, sku: str) -> CatalogItem | None:
        return self._by_sku.get(sku)

    def all(self) -> list[CatalogItem]:
        return list(self._by_sku.values())

    def find_candidates(
        self, query: str, k: int = 5, suppliers: list[str] | None = None
    ) -> list[ResolutionCandidate]:
        # Scope to the chosen tenants when asked (a list value means "supplier in
        # set"). FAISS filtering is post-retrieval, so over-fetch (fetch_k = corpus
        # size) to keep the scoped top-k exact at this scale; a real vector DB would
        # pre-filter and drop the over-fetch (Phase 1).
        flt = {"supplier": list(suppliers)} if suppliers else None
        # Cosine relevance can dip below 0 for dissimilar items; that's expected
        # (we clamp), so silence LangChain's out-of-[0,1] warning rather than spam it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            hits = self._store.similarity_search_with_relevance_scores(
                query, k=k, filter=flt, fetch_k=max(len(self._by_key), k)
            )
        candidates: list[ResolutionCandidate] = []
        for document, score in hits:
            key = (document.metadata.get("supplier"), document.metadata.get("sku"))
            item = self._by_key.get(key)
            if item is not None:
                candidates.append(ResolutionCandidate(item=item, score=_clamp01(score)))
        return candidates