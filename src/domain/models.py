"""Domain models — Pydantic ``BaseModel``, zero LLM/LangChain imports.

These are the types every layer speaks. Keeping this module dependency-light
means it imports anywhere without pulling in a tower of SDKs, and lets the
deterministic core be tested with plain object construction.

Only the catalog/RAG types live here so far. The order-pipeline types
(``LineItem``, ``CartOp``, supplier results, ``OrderState`` …) are reintroduced
test-first as we build outward from the UX — see ``_deferred.py`` for the
preserved design.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    """One SKU in the product catalog — the RAG knowledge base unit and the
    source of truth for *static* business rules (case packs, minimums, lids).

    Deliberately carries **no price or stock**: those are dynamic and change
    constantly, so they're served by the (mocked) supplier API at runtime —
    never embedded in the RAG corpus.
    """

    sku: str
    product_name: str
    aliases: list[str] = Field(default_factory=list)
    category: str
    unit_size: str  # human-readable, e.g. "16oz" or "case of 500"
    case_pack: int  # units per case
    min_order: int = 1  # minimum number of cases per order
    requires_lids: bool = False
    supplier: str  # supplier-keyed from day 1 (Stage-2 seam)


class ResolutionCandidate(BaseModel):
    """A catalog match with its retrieval similarity score.

    This is the shape the (deferred) FAISS retriever adapter emits — an
    anti-corruption boundary so SKU resolution never sees a LangChain
    ``Document``. Tests construct these directly, so no embeddings are needed.
    """

    item: CatalogItem
    score: float = Field(ge=0.0, le=1.0)