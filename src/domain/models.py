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
    # SKUs this item should be paired with (e.g. a container -> its lid). Drives
    # the generic "needs matching X — add them?" upsell; empty means no companion.
    # Data, not code: a new pairing is a catalog edit, never a code branch.
    companion_skus: list[str] = Field(default_factory=list)
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

    ``validate_rules`` sets the static-catalog flags (``NEEDS_COMPANION``,
    ``BELOW_MINIMUM``, ``ROUNDED_TO_CASE_PACK``); ``OUT_OF_STOCK`` is set by the
    inventory check against the supplier API. Some flags are *blocking* (the gate
    clarifies on them even at high confidence); ``ROUNDED_TO_CASE_PACK`` is
    informational only. The blocking set lives in ``src.domain.policies`` so
    the gate owns that policy, not the domain.
    """

    NEEDS_COMPANION = "needs_companion"
    OUT_OF_STOCK = "out_of_stock"
    AMBIGUOUS_SIZE = "ambiguous_size"
    BELOW_MINIMUM = "below_minimum"
    ROUNDED_TO_CASE_PACK = "rounded_to_case_pack"


class Companion(BaseModel):
    """A suggested add-on for a line: the companion's SKU plus its display name.

    Carried on the parent line (set during validation from the catalog) so the
    clarifying question can name it and a later "yes" can be mapped back to the
    exact SKU — no fuzzy re-resolution of the user's reply.
    """

    sku: str
    product_name: str


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
    # Suggested add-ons for this line (set by validation from the catalog's
    # companion_skus, minus any already in the cart). Drives the "needs matching
    # X — add them?" question and lets a later "yes" resolve to the exact SKU.
    companions: list[Companion] = Field(default_factory=list)


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


class InventoryStatus(BaseModel):
    """Dynamic stock picture for one SKU, from the supplier API (not the catalog).

    ``check_inventory`` sets ``OUT_OF_STOCK`` when ``in_stock`` is false;
    ``lead_time_days`` is the restock ETA the clarification can quote.
    """

    in_stock: bool
    quantity_on_hand: int = 0
    lead_time_days: int = 0


class OrderConfirmation(BaseModel):
    """The supplier's acknowledgement of a submitted order."""

    order_id: str
    supplier: str
    total: float | None = None


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