"""handle_turn — the UX boundary the Gradio callback delegates to.

Outside-in: these specs describe what the front-end needs from a turn. The UI
owns no logic — the handler turns the agent's structured result into a
renderable TurnResult (chat reply + cart) and, critically, never blanks the cart
on a question turn (spec §11). The inner agent is stubbed; its contract
(OrderAgent / AgentResult) is defined by exactly what the handler calls.
"""

from src.app.turn import TurnResult, handle_turn
from src.domain.cart import Cart
from src.domain.models import LineItem
from src.ports.order_agent import AgentResult


class FakeOrderAgent:
    """Stub agent: returns a preset AgentResult and records how it was called."""

    def __init__(self, result: AgentResult):
        self._result = result
        self.calls: list[tuple[str, Cart]] = []

    def run(self, message: str, cart: Cart) -> AgentResult:
        self.calls.append((message, cart))
        return self._result


def _deli_line(quantity: int = 3) -> LineItem:
    return LineItem(
        sku="DELI-16",
        product_name="16oz Deli Container",
        supplier="acme-foodservice",
        quantity=quantity,
    )


def test_handle_turn_delegates_message_and_cart_to_agent():
    cart = Cart()
    agent = FakeOrderAgent(AgentResult(draft_cart=cart))
    handle_turn("2 cases of straws", cart, agent)
    assert agent.calls == [("2 cases of straws", Cart())]


def test_returns_a_turn_result():
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart()))
    assert isinstance(handle_turn("hi", Cart(), agent), TurnResult)


def test_question_path_returns_answer_and_preserves_cart():
    cart = Cart(by_supplier={"acme-foodservice": [_deli_line(2)]})
    agent = FakeOrderAgent(AgentResult(draft_cart=cart, answer="Each case has 500 units."))
    result = handle_turn("how many per case?", cart, agent)
    assert result.reply == "Each case has 500 units."
    assert result.cart == cart  # §11: a question turn must not blank the cart panel


def test_clarification_path_surfaces_the_question():
    agent = FakeOrderAgent(
        AgentResult(
            draft_cart=Cart(),
            clarifications=["What size deli containers — 8, 16, or 32oz?"],
        )
    )
    result = handle_turn("I need deli containers", Cart(), agent)
    assert "What size deli containers" in result.reply
    assert result.cart == Cart()


def test_order_path_mirrors_cart_and_summarizes_it():
    new_cart = Cart(by_supplier={"acme-foodservice": [_deli_line(3)]})
    agent = FakeOrderAgent(AgentResult(draft_cart=new_cart))
    result = handle_turn("3 cases of 16oz deli", Cart(), agent)
    assert result.cart == new_cart
    assert "16oz Deli Container" in result.reply
    assert "3" in result.reply


def test_empty_cart_still_yields_a_nonempty_reply():
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart()))
    result = handle_turn("hello", Cart(), agent)
    assert result.cart == Cart()
    assert result.reply.strip()