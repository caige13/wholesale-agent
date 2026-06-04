"""Gradio front-end for the AI Order Desk — a chat window beside a live cart.

This package is the front-end split along its seams (it owns no business logic —
spec §11):

- :mod:`.seam` — the gradio-free callback core (``run_turn``) + session helpers
- :mod:`.cart_ops` — direct cart edits from the panel (remove / change quantity)
- :mod:`.render` — cart → packing-slip HTML (the read-only / print view)
- :mod:`.assets` — static UI assets (CSS, masthead, fonts, a11y JS)
- :mod:`.app` — the Gradio Blocks shell, callbacks, theme, and ``launch``

The testable seam (:func:`render_cart`, :func:`run_turn` and the cart-edit helpers)
imports no gradio, so it stays unit-testable keyless under ``make setup``;
``build_app``/``launch`` import gradio lazily.
"""

from .app import build_app, launch
from .cart_ops import remove_line, set_line_quantity, step_line_quantity
from .render import render_cart
from .seam import run_turn

__all__ = [
    "build_app",
    "launch",
    "remove_line",
    "render_cart",
    "run_turn",
    "set_line_quantity",
    "step_line_quantity",
]