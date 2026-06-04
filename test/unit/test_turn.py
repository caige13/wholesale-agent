"""handle_turn — the UX boundary the Gradio callback delegates to.

Outside-in: these specs describe what the front-end needs from a turn. The UI
owns no logic — the handler turns the agent's structured result into a
renderable TurnResult (chat reply + cart) and, critically, never blanks the cart
on a question turn (spec §11). The inner agent is stubbed; its contract
(OrderAgent / AgentResult) is defined by exactly what the handler calls.
"""

from src.app.turn import TurnResult, handle_turn
from src.domain.cart import Cart
from src.domain.models import Handoff, LineItem, OrderConfirmation
from src.ports.order_agent import AgentResult
from test.fakes import FakeOrderAgent


def _deli_line(quantity: int = 3) -> LineItem:
    return LineItem(
        sku="DELI-16",
        product_name="16oz Deli Container",
        supplier="acme-foodservice",
        quantity=quantity,
    )


def test_delegates_the_message_and_cart_to_the_agent():
    cart = Cart()
    agent = FakeOrderAgent(AgentResult(draft_cart=cart))
    handle_turn("2 cases of straws", cart, agent)
    assert agent.calls == [("2 cases of straws", Cart(), None)]


def test_returns_a_turn_result():
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart()))
    assert isinstance(handle_turn("hi", Cart(), agent), TurnResult)


def test_passes_chat_history_through_to_the_agent():
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart()))
    history = [
        {"role": "user", "content": "deli containers"},
        {"role": "assistant", "content": "which size?"},
    ]
    handle_turn("16oz", Cart(), agent, history=history)
    assert agent.last_history == history


def test_returns_the_answer_and_preserves_the_cart_on_a_question_turn():
    cart = Cart(by_supplier={"acme-foodservice": [_deli_line(2)]})
    agent = FakeOrderAgent(AgentResult(draft_cart=cart, answer="Each case has 500 units."))
    result = handle_turn("how many per case?", cart, agent)
    assert result.reply == "Each case has 500 units."
    assert result.cart == cart  # §11: a question turn must not blank the cart panel


def test_surfaces_the_question_on_a_clarification_turn():
    agent = FakeOrderAgent(
        AgentResult(
            draft_cart=Cart(),
            clarifications=["What size deli containers — 8, 16, or 32oz?"],
        )
    )
    result = handle_turn("I need deli containers", Cart(), agent)
    assert "What size deli containers" in result.reply
    assert result.cart == Cart()


def test_mirrors_the_cart_and_summarizes_it_on_an_order_turn():
    new_cart = Cart(by_supplier={"acme-foodservice": [_deli_line(3)]})
    agent = FakeOrderAgent(AgentResult(draft_cart=new_cart))
    result = handle_turn("3 cases of 16oz deli", Cart(), agent)
    assert result.cart == new_cart
    assert "16oz Deli Container" in result.reply
    assert "3" in result.reply


def test_yields_a_nonempty_reply_even_when_the_cart_is_empty():
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart()))
    result = handle_turn("hello", Cart(), agent)
    assert result.cart == Cart()
    assert result.reply.strip()


def test_whitespace_only_answer_falls_through_instead_of_blanking_reply():
    # A whitespace-only string is truthy in Python, so a blank answer would sail
    # past `if result.answer` and become an empty chat bubble. It must be treated
    # as "no answer" and fall through to the cart summary instead.
    cart = Cart(by_supplier={"acme-foodservice": [_deli_line(3)]})
    agent = FakeOrderAgent(AgentResult(draft_cart=cart, answer="  \n "))
    result = handle_turn("how many per case?", cart, agent)
    assert result.reply.strip()  # never a blank bubble
    assert "16oz Deli Container" in result.reply  # blank answer -> shows the cart


def test_summarizes_a_partial_line_without_printing_none():
    # An unresolved line (no product_name, no quantity) must not render as the
    # literal "None × None"; it falls back to the sku as a label.
    cart = Cart(by_supplier={"acme-foodservice": [LineItem(sku="DELI-16")]})
    agent = FakeOrderAgent(AgentResult(draft_cart=cart))
    result = handle_turn("...", cart, agent)
    assert "None" not in result.reply
    assert "DELI-16" in result.reply


def test_invites_checkout_on_an_unplaced_draft():
    # A non-empty draft with no confirmation should nudge toward placing the order.
    cart = Cart(by_supplier={"acme-foodservice": [_deli_line(3)]})
    agent = FakeOrderAgent(AgentResult(draft_cart=cart))  # no confirmation
    result = handle_turn("add deli", Cart(), agent)
    assert "place the order" in result.reply
    assert "Order confirmed" not in result.reply


def test_renders_a_human_handoff_with_its_ticket_reference_deterministically():
    # The intent-path handoff is composed by us, not the LLM, so the ticket + callback
    # are always present (the bug was the QA model paraphrasing and dropping the ticket).
    handoff = Handoff(ticket_id="SUP-AB12CD", reason="x", callback_number="1-800-555-0000")
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart(), handoff=handoff))
    result = handle_turn("get me a person", Cart(), agent)
    assert "SUP-AB12CD" in result.reply
    assert "1-800-555-0000" in result.reply


def test_shows_the_confirmation_when_the_order_is_placed():
    cart = Cart(by_supplier={"acme-foodservice": [_deli_line(3)]})
    confirmation = OrderConfirmation(order_id="ACME-1", supplier="acme-foodservice")
    agent = FakeOrderAgent(AgentResult(draft_cart=cart, confirmation=confirmation))
    result = handle_turn("place it", cart, agent)
    assert "Order confirmed: ACME-1" in result.reply
    assert "place the order?" not in result.reply