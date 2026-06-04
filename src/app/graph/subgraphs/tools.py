"""Read-only LangChain tools the QA agent can call.

The QUESTION path is a real tool-calling loop (``bind_tools`` -> ``ToolNode`` ->
``ToolMessage``), so the model decides *which* lookup to run and *when*. These three
tools are the model-callable face of the same ports the deterministic order pipeline
uses — ``search_catalog`` over ``CatalogRepository`` (RAG retrieval) and
``check_inventory`` / ``get_price`` over ``SupplierGateway``.

They are deliberately **read-only**: the order-write path (apply, the clarify/draft
gate, ``submit_order``) stays deterministic and is never exposed as a tool, so the
model can answer questions but can never silently mutate or place an order.

Each tool closes over the injected adapter (same dependency-injection discipline as
the graph nodes) and returns **text** — the format the model reads off a ToolMessage.
The deterministic nodes keep calling the typed port methods directly; these wrappers
add no logic, only a model-facing rendering of the same data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.ports import CatalogRepository, EscalationGateway, SupplierGateway


class _SearchCatalogArgs(BaseModel):
    query: str = Field(description="Free-text product description, e.g. '16oz deli container'.")
    k: int = Field(default=3, description="How many candidate products to return (best first).")


class _SkuArgs(BaseModel):
    sku: str = Field(description="The exact catalog SKU, e.g. 'DELI-16'.")


class _EscalateArgs(BaseModel):
    reason: str = Field(
        description=(
            "A brief reason the question needs a human — what the customer asked that "
            "the read-only lookups can't answer (e.g. 'wants to return a case')."
        )
    )


def build_order_desk_tools(catalog: CatalogRepository, supplier: SupplierGateway):
    """Build the read-only tools for the QA agent, bound to the injected adapters."""
    from langchain_core.tools import StructuredTool

    def search_catalog(query: str, k: int = 3) -> str:
        """Find catalog products matching a free-text description."""
        candidates = catalog.find_candidates(query, k=k)
        if not candidates:
            return "(no matching catalog entries)"
        return "\n".join(
            f"- {c.item.product_name} (sku={c.item.sku}, {c.item.unit_size}, "
            f"{c.item.case_pack} per case, min {c.item.min_order} case(s))"
            for c in candidates
        )

    def check_inventory(sku: str) -> str:
        """Check live stock for a SKU (in-stock, quantity on hand, restock lead time)."""
        status = supplier.check_inventory(sku)
        if not status.in_stock:
            eta = f", restock in {status.lead_time_days} day(s)" if status.lead_time_days else ""
            return f"{sku}: out of stock{eta}"
        return f"{sku}: in stock ({status.quantity_on_hand} on hand)"

    def get_price(sku: str) -> str:
        """Get the unit price for a SKU."""
        price = supplier.get_price(sku)
        return f"{sku}: not priced" if price is None else f"{sku}: ${price:.2f} per case"

    return [
        StructuredTool.from_function(
            func=search_catalog,
            name="search_catalog",
            description=(
                "Search the product catalog for items matching a free-text description. "
                "Use this first to find a product and its SKU before checking stock or price."
            ),
            args_schema=_SearchCatalogArgs,
        ),
        StructuredTool.from_function(
            func=check_inventory,
            name="check_inventory",
            description=(
                "Check whether a SKU is in stock and how many units are on hand. "
                "Stock is dynamic supplier data, not in the catalog — always use this tool for it."
            ),
            args_schema=_SkuArgs,
        ),
        StructuredTool.from_function(
            func=get_price,
            name="get_price",
            description=(
                "Get the unit (per-case) price for a SKU. Price is dynamic supplier data, "
                "not in the catalog — always use this tool for it."
            ),
            args_schema=_SkuArgs,
        ),
    ]


def build_escalation_tool(escalation: EscalationGateway):
    """Build the model-callable handoff tool, bound to the escalation gateway.

    Kept separate from the read-only data tools: it's a *write* (it opens a support
    ticket), and it's only added to the QA agent's toolset when an escalation gateway
    is wired, so the read-only tool set stays exactly three. The model calls this when
    its lookups can't answer — turning the old "I don't have that information" dead-end
    into a real human handoff. Returns text (the ticket) the model relays to the user.
    """
    from langchain_core.tools import StructuredTool

    def escalate_to_human(reason: str) -> str:
        """Hand the conversation off to a human specialist when the tools can't help."""
        handoff = escalation.create_handoff(reason=reason, summary=reason)
        return (
            f"Escalation opened. Give the customer their reference, ticket {handoff.ticket_id}, "
            f"and tell them a specialist will follow up within about {handoff.eta_minutes} "
            f"minutes and that they can reach our team directly at {handoff.callback_number}."
        )

    return [
        StructuredTool.from_function(
            func=escalate_to_human,
            name="escalate_to_human",
            description=(
                "Hand the conversation off to a human specialist. Call this when the "
                "customer's request is outside product stock/pricing (returns, billing, "
                "account changes, complaints) or the catalog and tools simply can't answer "
                "it — instead of guessing or saying you don't know."
            ),
            args_schema=_EscalateArgs,
        )
    ]
