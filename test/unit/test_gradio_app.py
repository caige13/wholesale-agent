"""The UI seam — what the Gradio callback delegates to (render + handler).

TDD, outside-in: these specs pin the two things the front-end needs that aren't
Gradio internals. ``render_cart`` turns the cart into the packing-slip markup; the
panel must never print a literal "None" and must survive a half-resolved line.
``run_turn`` is the callback's core — it appends the turn to the chat history and,
critically, returns ``result.cart`` so a question turn never blanks the panel
(spec §11). The agent is stubbed; gradio is never imported here (the seam is
dependency-light by design, so it's testable keyless under ``make setup``).
"""

from src.domain.cart import Cart
from src.domain.models import Flag, LineItem
from src.interfaces.gradio_app import (
    remove_line,
    render_cart,
    run_turn,
    set_line_quantity,
    step_line_quantity,
)
from src.ports.order_agent import AgentResult
from test.fakes import FakeOrderAgent

SUPPLIER = "acme-foodservice"


def _deli_line(quantity: int = 3, **kw) -> LineItem:
    return LineItem(
        sku="DELI-16",
        product_name="16oz Deli Container",
        supplier=SUPPLIER,
        quantity=quantity,
        **kw,
    )


def _cart_with(*lines: LineItem) -> Cart:
    by_supplier: dict[str, list[LineItem]] = {}
    for line in lines:
        by_supplier.setdefault(line.supplier, []).append(line)
    return Cart(by_supplier=by_supplier)


# --- run_turn -------------------------------------------------------------


def test_appends_the_user_message_and_the_reply_in_messages_format():
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart(), answer="hello there"))
    history, _ = run_turn(agent, "hi", [], Cart())
    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]


def test_keeps_earlier_turns_when_appending_a_new_one():
    prior = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart(), answer="second reply"))
    history, _ = run_turn(agent, "second", prior, Cart())
    assert history[:2] == prior
    assert history[-1] == {"role": "assistant", "content": "second reply"}


def test_returns_the_unchanged_cart_on_a_question_turn():
    # §11: a question turn answers from RAG and must hand the cart straight back,
    # or the panel blanks. We always render result.cart, so this is the guarantee.
    cart = _cart_with(_deli_line(2))
    agent = FakeOrderAgent(AgentResult(draft_cart=cart, answer="Each case has 500."))
    _, returned = run_turn(agent, "how many per case?", [], cart)
    assert returned == cart


def test_forwards_the_prior_conversation_history_to_the_agent():
    # Follow-ups ("make it 3") only make sense with the prior turns in hand, so the
    # UI must hand the conversation so far to the agent.
    prior = [
        {"role": "user", "content": "3 cases of deli"},
        {"role": "assistant", "content": "Added 3 cases."},
    ]
    agent = FakeOrderAgent(AgentResult(draft_cart=Cart()))
    run_turn(agent, "make it 5", prior, Cart())
    _, _, forwarded_history = agent.calls[-1]
    assert forwarded_history == prior


def test_returns_the_agents_updated_cart_on_an_order_turn():
    new_cart = _cart_with(_deli_line(3))
    agent = FakeOrderAgent(AgentResult(draft_cart=new_cart))
    _, returned = run_turn(agent, "3 cases of deli", [], Cart())
    assert returned == new_cart


# --- render_cart ----------------------------------------------------------


def test_shows_the_awaiting_order_stamp_when_the_cart_is_empty():
    html = render_cart(Cart())
    assert "Awaiting Order" in html  # uppercased visually by CSS, title-case in markup


def test_groups_lines_under_their_supplier():
    html = render_cart(_cart_with(_deli_line(3)))
    assert SUPPLIER in html
    assert "16oz Deli Container" in html
    assert "3" in html
    assert "DELI-16" in html


def test_does_not_print_none_for_a_half_resolved_line():
    # An unresolved line (no product_name, no quantity) must not surface as the
    # literal "None" — it falls back to the sku as the label, like turn._render_line.
    html = render_cart(_cart_with(LineItem(sku="DELI-16", supplier=SUPPLIER)))
    assert "None" not in html
    assert "DELI-16" in html


def test_renders_a_unit_price_when_present():
    html = render_cart(_cart_with(_deli_line(3, unit_price=12.5)))
    assert "12.50" in html


def test_renders_a_flag_chip_for_a_flagged_line():
    line = _deli_line(3, flags=[Flag.NEEDS_COMPANION])
    html = render_cart(_cart_with(line))
    assert "needs companion" in html


def test_never_raises_on_a_cart_with_missing_fields():
    # Defensive: a line with almost nothing set still renders without blowing up.
    render_cart(_cart_with(LineItem(supplier=SUPPLIER)))


# --- cart editing (remove / set quantity) ---------------------------------
# These route through the domain's one mutation point (Cart.apply); the UI only
# declares the op, it never mutates the cart itself (spec §11).


def _pcup_line(quantity: int = 2) -> LineItem:
    return LineItem(
        sku="PCUP-4", product_name="4oz Portion Cup", supplier=SUPPLIER, quantity=quantity
    )


def test_remove_line_drops_only_the_matching_line():
    out = remove_line(_cart_with(_deli_line(3), _pcup_line(2)), SUPPLIER, "DELI-16")
    skus = [line.sku for line in out.all_lines()]
    assert skus == ["PCUP-4"]


def test_remove_line_empties_the_cart_when_it_was_the_last_line():
    assert remove_line(_cart_with(_deli_line(3)), SUPPLIER, "DELI-16").is_empty()


def test_set_line_quantity_replaces_the_quantity_without_duplicating_the_line():
    out = set_line_quantity(_cart_with(_deli_line(2)), SUPPLIER, "DELI-16", 5)
    lines = out.by_supplier[SUPPLIER]
    assert len(lines) == 1
    assert lines[0].quantity == 5


def test_set_line_quantity_to_zero_or_less_removes_the_line():
    assert set_line_quantity(_cart_with(_deli_line(2)), SUPPLIER, "DELI-16", 0).is_empty()
    assert set_line_quantity(_cart_with(_deli_line(2)), SUPPLIER, "DELI-16", -1).is_empty()


def test_editing_returns_a_new_cart_and_leaves_the_original_untouched():
    cart = _cart_with(_deli_line(3))
    set_line_quantity(cart, SUPPLIER, "DELI-16", 9)
    remove_line(cart, SUPPLIER, "DELI-16")
    assert cart.by_supplier[SUPPLIER][0].quantity == 3  # original unchanged


# --- step_line_quantity (the panel's −/+ stepper) -------------------------
# step_line_quantity reads the *live* cart rather than a quantity captured when the
# panel rendered, so a burst of −/+ clicks before the @gr.render re-render still
# steps from the true current value.


def test_step_line_quantity_increments_from_the_live_cart_value():
    out = step_line_quantity(_cart_with(_deli_line(3)), SUPPLIER, "DELI-16", +1)
    assert out.by_supplier[SUPPLIER][0].quantity == 4


def test_step_line_quantity_decrements_from_the_live_cart_value():
    out = step_line_quantity(_cart_with(_deli_line(3)), SUPPLIER, "DELI-16", -1)
    assert out.by_supplier[SUPPLIER][0].quantity == 2


def test_step_line_quantity_removes_the_line_when_stepping_below_one():
    assert step_line_quantity(_cart_with(_deli_line(1)), SUPPLIER, "DELI-16", -1).is_empty()


# --- chip classification --------------------------------------------------
# The info/warn split is owned by policies.BLOCKING_FLAGS, not a local copy.


def test_renders_an_informational_chip_for_a_rounded_to_case_pack_flag():
    # ROUNDED_TO_CASE_PACK is the one non-blocking flag, so it renders as info,
    # never as a warning.
    html = render_cart(_cart_with(_deli_line(3, flags=[Flag.ROUNDED_TO_CASE_PACK])))
    assert "chip--info" in html
    assert "chip--warn" not in html