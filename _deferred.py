"""PRESERVED SCAFFOLDING — intentionally NOT committed (see .gitignore).

This file is a holding pen for designed-but-not-yet-needed code so we can build
the repo strictly test-first (outside-in from the UX) without losing the design.
Each block is lifted back into the real package the moment a failing test /
the front-end interface actually requires it. Nothing here is imported by the
application or tests.

Order of expected reintroduction (outside-in):
  1. UI / cart interface  -> LineItem, CartOp, CartOpKind, OrderState
  2. supplier tools        -> SupplierGateway, InventoryStatus, OrderConfirmation
  3. intent routing        -> Intent, OrderStatus, Flag (some flags arrive with
                              validate_rules / inventory check)
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Protocol, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Intent(StrEnum):
    ORDER = "order"
    REORDER = "reorder"
    QUESTION = "question"


class OrderStatus(StrEnum):
    PARSING = "parsing"
    NEEDS_CLARIFICATION = "needs_clarification"
    DRAFTED = "drafted"
    CONFIRMED = "confirmed"
    SUBMITTED = "submitted"


class CartOpKind(StrEnum):
    ADD = "add"
    SET_QUANTITY = "set_quantity"
    REMOVE = "remove"


class Flag(StrEnum):
    NEEDS_LIDS = "needs_lids"
    OUT_OF_STOCK = "out_of_stock"
    AMBIGUOUS_SIZE = "ambiguous_size"
    BELOW_MINIMUM = "below_minimum"
    ROUNDED_TO_CASE_PACK = "rounded_to_case_pack"


# ---------------------------------------------------------------------------
# Order-pipeline models
# ---------------------------------------------------------------------------
class LineItem(BaseModel):
    raw_text: str
    sku: str | None = None
    product_name: str | None = None
    supplier: str | None = None
    unit: str | None = None
    quantity: int | None = None
    unit_quantity: int | None = None  # raw unit count; validate_rules rounds -> cases
    unit_price: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    flags: list[Flag] = Field(default_factory=list)


class CartOp(BaseModel):
    op: CartOpKind
    item: LineItem


class InventoryStatus(BaseModel):
    in_stock: bool
    quantity_on_hand: int = 0
    lead_time_days: int = 0


class OrderConfirmation(BaseModel):
    order_id: str
    supplier: str
    total: float | None = None


# ---------------------------------------------------------------------------
# Graph state (LangGraph TypedDict + reducers)
# ---------------------------------------------------------------------------
class OrderState(TypedDict, total=False):
    restaurant_id: str
    raw_message: str
    clean_message: str
    pii_found: list[str]
    intent: Intent | None
    line_items: list[LineItem]
    cart_ops: list[CartOp]
    draft_cart: dict[str, list[LineItem]]  # {supplier: [LineItem]} — persists across turns
    clarifications: Annotated[list[str], add]
    answer: str | None
    status: OrderStatus
    confirmation_ids: dict


# ---------------------------------------------------------------------------
# Supplier gateway port (mocked supplier API: price / inventory / submit)
# ---------------------------------------------------------------------------
class SupplierGateway(Protocol):
    supplier: str

    def get_price(self, sku: str) -> float | None: ...

    def check_inventory(self, sku: str) -> InventoryStatus: ...

    def submit_order(self, items: list[LineItem]) -> OrderConfirmation: ...