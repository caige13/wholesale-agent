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

from enum import StrEnum

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
    case_pack: int = Field(gt=0)  # units per case; a divisor in validate_rules
    min_order: int = Field(default=1, gt=0)  # minimum number of cases per order
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


class Flag(StrEnum):
    """Per-line conditions raised during the pipeline.

    ``validate_rules`` sets the static-catalog flags (``NEEDS_LIDS``,
    ``BELOW_MINIMUM``, ``ROUNDED_TO_CASE_PACK``); ``OUT_OF_STOCK`` is set by the
    inventory check against the supplier API. Some flags are *blocking* (the gate
    clarifies on them even at high confidence); ``ROUNDED_TO_CASE_PACK`` is
    informational only. The blocking set lives in ``src.domain.policies`` so
    the gate owns that policy, not the domain.
    """

    NEEDS_LIDS = "needs_lids"
    OUT_OF_STOCK = "out_of_stock"
    AMBIGUOUS_SIZE = "ambiguous_size"
    BELOW_MINIMUM = "below_minimum"
    ROUNDED_TO_CASE_PACK = "rounded_to_case_pack"


class LineItem(BaseModel):
    """A line in the order / running cart.

    Fields fill in as the line moves through the pipeline: display fields from
    parsing/resolution, ``confidence`` from resolution (drives the gate), and
    ``flags`` from validation / inventory. ``unit_quantity`` returns with
    validate_rules — see ``_deferred.py``.
    """

    raw_text: str = ""  # the phrase parsed from the message; resolve_skus matches on it
    sku: str | None = None
    product_name: str | None = None
    supplier: str | None = None
    unit: str | None = None
    quantity: int | None = None
    # Raw unit count when the user orders in units ("1000 containers") rather than
    # cases; validate_rules rounds it up into `quantity` (whole cases).
    unit_quantity: int | None = None
    unit_price: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    flags: list[Flag] = Field(default_factory=list)
    # When resolution is ambiguous, the candidate product names to offer the user
    # ("did you mean 8oz, 16oz, or 32oz?"). Empty once resolved.
    options: list[str] = Field(default_factory=list)


class CartOpKind(StrEnum):
    """The three mutations the agent can apply to the running cart."""

    ADD = "add"
    SET_QUANTITY = "set_quantity"
    REMOVE = "remove"


class CartOp(BaseModel):
    """One mutation against the cart. ``item``'s (supplier, sku) identifies the
    target line for set_quantity / remove.
    """

    op: CartOpKind
    item: LineItem


class Intent(StrEnum):
    """What the user is doing this turn — routes the graph after redaction.

    Extensible: v2 adds RETURN / ORDER_STATUS without touching callers.
    """

    ORDER = "order"
    REORDER = "reorder"
    QUESTION = "question"


class OrderStatus(StrEnum):
    """Lifecycle of the turn's order — the terminal status the UI/trace reads."""

    PARSING = "parsing"
    NEEDS_CLARIFICATION = "needs_clarification"
    DRAFTED = "drafted"
    CONFIRMED = "confirmed"
    SUBMITTED = "submitted"