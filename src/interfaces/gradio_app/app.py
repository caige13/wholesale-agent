"""The Gradio shell — Blocks layout, callbacks, theme, and ``launch`` (lazy gradio).

The UI owns no business logic (spec §11): it threads the running cart through
``gr.State``, calls the seam (which delegates to ``handle_turn``/``stream_turn``),
and renders whatever cart comes back. Because we *always* render ``result.cart``,
an interleaved product question answers in chat without blanking the cart panel.

gradio lives only in the ``agent`` extra, so it's imported lazily inside
:func:`build_app` and :func:`launch` — that keeps the seam (:mod:`.seam`,
:mod:`.render`) unit-testable keyless under ``make setup``.
"""

from __future__ import annotations

import logging

from src.app.turn import handle_turn, stream_turn
from src.domain.cart import Cart
from src.ports.order_agent import OrderAgent

from .assets import _A11Y_JS, _CSS, _MASTHEAD
from .cart_ops import remove_line, step_line_quantity
from .render import (
    _EMPTY_SLIP_INNER,
    _footer_html,
    _group_header_html,
    _line_text_html,
    _slip_header_html,
)
from .seam import _EXAMPLES, _initial_history, _new_thread, _ui_trace


def _theme(gr):
    """The paper-stock theme — Hanken Grotesk body over a JetBrains Mono base."""
    return gr.themes.Base(
        font=[gr.themes.GoogleFont("Hanken Grotesk"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )


def build_app(agent: OrderAgent, *, trace_enabled: bool = False):
    """Build the Gradio Blocks app over ``agent``. Gradio imported lazily.

    Theme + css are applied at ``launch`` time (Gradio 6 moved them off the
    ``Blocks`` constructor), so the Blocks here is structural only.

    ``trace_enabled`` (set by ``launch`` from settings) decides whether each turn
    carries a ``ui`` ``TraceContext`` for LangSmith; tests build the app without it.
    """
    import gradio as gr

    def respond(message: str, history: list[dict], cart: Cart, thread_id: str):
        """Stream the reply as work happens, with silent recovery.

        Yields map to ``[chat, cart_state, msg]``. Progress shows as faint italic ink;
        the answer streams in; the cart commits once, on the final frame. The agent owns
        history via the checkpointer, so it isn't threaded here — ``history`` is just the
        chat widget's display. A stream error falls back to one blocking turn, unseen.
        """
        if not message or not message.strip():
            yield history, gr.skip(), ""
            return

        def bubble(text: str) -> list[dict]:
            return [*history,
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": text}]

        # Show the user's bubble and clear the input before the agent runs.
        yield bubble("…"), gr.skip(), ""

        trace = _ui_trace(len(history)) if trace_enabled else None
        try:
            done = False
            for frame in stream_turn(message, cart, agent, trace=trace, thread_id=thread_id):
                text = f"*{frame.reply}*" if frame.phase == "progress" else frame.reply
                yield bubble(text), (frame.cart if frame.done else gr.skip()), ""
                done = frame.done
            if done:
                return
        except Exception:  # noqa: BLE001 — a stream hiccup must never reach the user
            logging.getLogger(__name__).exception("streaming turn failed — recovering silently")

        # Silent recovery: one blocking turn, so a stream error stays invisible.
        result = handle_turn(message, cart, agent, trace=trace, thread_id=thread_id)
        yield bubble(result.reply), result.cart, gr.skip()

    def reset():
        return _initial_history(), Cart(), "", _new_thread()  # New ticket = fresh thread

    def cart_panel(cart: Cart):
        """Render the cart panel — read-only slip text plus an inline −/+/✕ control
        per line. Re-run by ``@gr.render`` whenever ``cart_state`` changes."""
        if cart.is_empty():
            # The panel column (#slippanel) is already the slip card, so render the
            # inner stamp/hint with only the .slip--empty centering wrapper —
            # _EMPTY_SLIP's full .slip wrapper would double the border (that form is
            # for the standalone render_cart view).
            gr.HTML(f'<div class="slip--empty">{_EMPTY_SLIP_INNER}</div>')
            return
        gr.HTML(_slip_header_html(cart))
        for supplier, items in cart.by_supplier.items():
            gr.HTML(_group_header_html(supplier))
            for item in items:
                editable = bool(item.sku)
                qty = item.quantity if item.quantity is not None else "—"
                with gr.Row(elem_classes="lnrow", equal_height=True):
                    # −/qty/+ are wrapped in their own Row so CSS can draw one
                    # connected "stepper" pill around them; ✕ stays a separate
                    # control. Buttons get no aria-label here (gr.Button has none
                    # in Gradio 6) — they're labelled by the demo.load MutationObserver
                    # (_A11Y_JS) once rendered.
                    with gr.Row(elem_classes="stepper"):
                        dec = gr.Button("−", elem_classes="qbtn", scale=0,
                                        min_width=0, interactive=editable)
                        gr.HTML(f'<div class="qval">{qty}</div>', elem_classes="qcell")
                        inc = gr.Button("+", elem_classes="qbtn", scale=0,
                                        min_width=0, interactive=editable)
                    gr.HTML(_line_text_html(item), elem_classes="lncell")
                    rm = gr.Button("✕", elem_classes="rmbtn", scale=0,
                                   min_width=30, interactive=editable)
                if editable:
                    s, k = item.supplier, item.sku
                    dec.click(lambda c, s=s, k=k: step_line_quantity(c, s, k, -1),
                              cart_state, cart_state)
                    inc.click(lambda c, s=s, k=k: step_line_quantity(c, s, k, +1),
                              cart_state, cart_state)
                    rm.click(lambda c, s=s, k=k: remove_line(c, s, k),
                             cart_state, cart_state)
        gr.HTML(_footer_html(cart))

    with gr.Blocks(title="The Order Desk", fill_height=True) as demo:
        cart_state = gr.State(Cart())
        thread_state = gr.State("default")  # checkpointer key; unique per session (demo.load)
        gr.HTML(_MASTHEAD)

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    value=_initial_history(),
                    elem_id="deskchat",
                    height=440,
                    show_label=False,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        elem_id="askbox",
                        placeholder="e.g. 3 cases of 16oz deli containers and some salsa cups",
                        show_label=False,
                        scale=8,
                        autofocus=True,
                    )
                    send = gr.Button("Send", elem_id="sendbtn", scale=1, min_width=90)
                with gr.Row():
                    new_ticket = gr.Button("＋ New ticket", elem_id="newbtn", scale=1)
                gr.Examples(examples=_EXAMPLES, inputs=msg, label="Try")

            with gr.Column(scale=2, elem_id="slippanel"):
                gr.render(inputs=[cart_state])(cart_panel)

        outputs = [chat, cart_state, msg]
        respond_inputs = [msg, chat, cart_state, thread_state]
        msg.submit(respond, respond_inputs, outputs)
        send.click(respond, respond_inputs, outputs)
        new_ticket.click(reset, None, [chat, cart_state, msg, thread_state])

        # Give each session its own checkpointer thread, and label the cart's glyph
        # buttons for screen readers on load (see _A11Y_JS).
        demo.load(_new_thread, None, thread_state)
        demo.load(None, None, None, js=_A11Y_JS)

    return demo


def launch(**kwargs):
    """Build the real agent once (Gemini + FAISS) and launch the styled UI."""
    import gradio as gr

    from src.bootstrap import build_agent
    from src.config import get_settings
    from src.observability import configure_logging, configure_tracing, tracing_status_line

    configure_logging()
    settings = get_settings()
    tracing = configure_tracing(settings)
    logging.getLogger(__name__).info(tracing_status_line(settings, tracing))

    kwargs.setdefault("theme", _theme(gr))
    kwargs.setdefault("css", _CSS)
    build_app(build_agent(), trace_enabled=tracing).launch(**kwargs)