"""LLM-backed graph nodes — intent classification and order parsing.

Each node takes the injected LangChain chat model (Gemini at runtime, a fake in
tests) and calls it; this module imports no LangChain itself, so the contract
tests stay keyless. The structured-output schemas below are what the model fills.

The QUESTION path is no longer here: it's a tool-calling subgraph that the model
drives (see ``subgraphs/qa_agent``), embedded directly in the root graph.

The probabilistic quality of these nodes (did it parse/classify correctly?) is
the eval set's job — the contract tests here only pin the state shape & routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.app.graph.nodes import pending_companions
from src.domain.cart import Cart
from src.domain.models import (
    CartOp,
    CartOpKind,
    Companion,
    Intent,
    LineItem,
)
from src.ports import CatalogRepository

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# --- Structured-output schemas the model fills ------------------------------
class ParsedItem(BaseModel):
    """One requested change to the order, extracted from the message."""

    phrase: str  # product phrase, typos corrected to standard wording ("16oz deli")
    quantity: int | None = None  # number of cases, if stated
    unit_quantity: int | None = None  # raw unit count, if stated ("1200 containers")
    # add a new/more of an item, set_quantity to change an existing line to an
    # exact amount ("make it 3"), or remove an existing line ("drop the limes").
    action: CartOpKind = CartOpKind.ADD


class AcceptedCompanion(BaseModel):
    """One offered add-on the user agreed to, picked from the pending offer."""

    name: str  # the offered companion's product name, copied verbatim from the offer
    # Cases, ONLY if the user stated an amount for this add-on ("just 2 cases of
    # lids"); left null otherwise so add_companions sizes it deterministically.
    quantity: int | None = None


class ParsedOrder(BaseModel):
    items: list[ParsedItem] = Field(default_factory=list)
    # The offered companion add-ons the user agreed to this turn — a closed-set
    # pick from the pending offer (NOT free text). add_companions maps each name
    # back to its exact SKU; empty means none accepted.
    accepted_companions: list[AcceptedCompanion] = Field(default_factory=list)
    # True only when the user signals they are finished and want the order placed
    # ("that's it", "place the order", "I'm done"). Drives submission; otherwise the
    # cart is kept as a running draft so they can keep adding across turns.
    place_order: bool = False


class IntentResult(BaseModel):
    intent: Intent


# --- Prompts ----------------------------------------------------------------
_INTENT_INSTRUCTIONS = (
    "Classify the restaurant's message as one of: 'order' (placing/adding items), "
    "'reorder' (repeat a usual order), 'question' (asking about a product), or "
    "'escalate'. "
    "A short reply that accepts or declines a pending add-on offer ('yes', 'yes "
    "please', 'sure', 'add them', 'no thanks') is an 'order', not a 'question'.\n"
    "Use 'escalate' ONLY when the user explicitly asks for a human/representative "
    "('let me talk to someone', 'can I get a person', 'this isn't working'), or asks "
    "for something the order desk cannot do — returns, refunds, disputes/chargebacks, "
    "billing or account changes, or cancelling a placed order. A normal product "
    "question the catalog might not cover is still 'question', NOT 'escalate'.\n"
    "Message:\n"
)
_PARSE_INSTRUCTIONS = (
    "Extract the changes the restaurant wants to make to its order. Include EVERY "
    "item the user asks for, even one you don't recognize as a catalog product — "
    "give its phrase as stated; never silently drop a requested item (the system "
    "will ask about anything it can't match). For each item give the product phrase, "
    "correcting obvious spelling/spacing typos to the catalog's standard wording "
    "(e.g. '16 ounze deli ocntainers' → '16oz deli container') — but do NOT add "
    "detail the user didn't give (if they didn't state a size, don't invent one). "
    "Also give either the number of cases (quantity) or a raw unit count "
    "(unit_quantity) if ordered in units.\n"
    "Choose an action per item:\n"
    "- 'add' for a new item or more of something.\n"
    "- 'set_quantity' when the item is ALREADY in the current cart and the user "
    "states a new total — e.g. 'make it 3', 'change the deli to 3 cases'. Do NOT "
    "use 'add' for these; set the quantity to the new total.\n"
    "- 'remove' to drop an item that's in the cart ('drop the limes').\n"
    "If the user's whole message is just an affirmation ('yes', 'yea', 'sure', 'go "
    "ahead', 'please do') and the assistant's most recent turn offered to place an "
    "order for a SPECIFIC product, treat it as an 'add' for that product: use the "
    "product wording from that offer, and the quantity it (or an earlier turn) "
    "stated — leave quantity null if none was stated and the system will ask. This "
    "is for a primary product the assistant proposed ordering; accepted add-on "
    "offers instead go in accepted_companions (below), never in items.\n\n"
    "Pending add-on offer (the assistant already proposed adding these alongside "
    "items in the cart):\n{offer}\n"
    "If the user agrees to some or all of these add-ons ('yes', 'yes please', "
    "'sure', 'add the lids but not the cups'), list each accepted one in "
    "accepted_companions with its name copied verbatim from the offer — and a "
    "quantity ONLY if the user states an amount for it ('just 2 cases of lids'); "
    "otherwise leave its quantity null. If the user declines or doesn't mention "
    "them, leave accepted_companions empty. Never re-add the parent item, and put "
    "accepted add-ons ONLY in accepted_companions, never in items.\n\n"
    "Set place_order to true ONLY when the user signals they are finished and want "
    "the order submitted ('that's it', 'place the order', 'I'm done', 'submit it', "
    "'check out'). Accepting an add-on offer ('yes please') is NOT placing the order. "
    "Adding/removing/changing items keeps place_order false unless they also say "
    "they're done (e.g. 'add napkins and that's it' → add napkins AND place_order=true)."
    "\n\nCurrent cart:\n{cart}\n\nMessage:\n{message}"
)
# --- Nodes (graph order: intent -> parse; the question branch is a subgraph) -
def intent_node(state: dict, model: BaseChatModel) -> dict:
    prompt = _format_history(state.get("history")) + _INTENT_INSTRUCTIONS + state["clean_message"]
    result = model.with_structured_output(IntentResult).invoke(prompt)
    return {"intent": result.intent}


def parse_node(state: dict, model: BaseChatModel, catalog: CatalogRepository) -> dict:
    pending = pending_companions(state.get("draft_cart") or Cart(), catalog)
    prompt = _format_history(state.get("history")) + _PARSE_INSTRUCTIONS.format(
        cart=_cart_context(state.get("draft_cart")),
        offer=_offer_context(pending),
        message=state["clean_message"],
    )
    parsed = model.with_structured_output(ParsedOrder).invoke(prompt)
    cart_ops = [
        CartOp(
            op=item.action,
            item=LineItem(
                raw_text=item.phrase, quantity=item.quantity, unit_quantity=item.unit_quantity
            ),
        )
        for item in parsed.items
    ]
    accepted = [{"name": a.name, "quantity": a.quantity} for a in parsed.accepted_companions]
    return {
        "cart_ops": cart_ops,
        "accepted_companions": accepted,
        "place_order": parsed.place_order,
    }


def _offer_context(pending: list[Companion]) -> str:
    """Render the still-open add-on offer so the model picks from a closed set."""
    if not pending:
        return "(none)"
    return "\n".join(f"- {companion.product_name}" for companion in pending)


def _format_history(history: list[dict] | None) -> str:
    """Render the last few turns so the model can interpret a follow-up in context."""
    if not history:
        return ""
    lines = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history[-6:]
    )
    return f"Recent conversation:\n{lines}\n\n"


def _cart_context(cart) -> str:
    """Render the current cart so the model can resolve set_quantity / remove."""
    if cart is None or cart.is_empty():
        return "(cart is empty)"
    return "\n".join(
        f"- {item.quantity} x {item.product_name} ({item.sku})" for item in cart.all_lines()
    )