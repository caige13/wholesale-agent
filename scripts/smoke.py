"""Manual smoke test — drive the real agent (Gemini + FAISS) across a few turns.

Not part of the test suite: this hits the real LLM and is for eyeballing that the
whole vertical slice works end-to-end and that traces show up in LangSmith.

Run:  uv run python scripts/smoke.py
Needs: GOOGLE_API_KEY in .env (set LANGSMITH_TRACING=true + LANGSMITH_API_KEY to
capture a trace). First run downloads the embedding model + builds the index.
"""

import logging
import sys
from pathlib import Path

# Make `import src...` work when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.turn import handle_turn  # noqa: E402
from src.bootstrap import build_agent  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.domain.cart import Cart  # noqa: E402
from src.observability import (  # noqa: E402
    TraceContext,
    configure_logging,
    configure_tracing,
    tracing_status_line,
)

# A conversation that exercises every path: clean order, alias, a PII guardrail,
# an ambiguous item (should clarify), and an interleaved question (cart unchanged).
TURNS = [
    "I need 2 cases of wrapped straws",
    "also a case of salsa cups",
    "deliver to 555-123-4567 and add 3 cases of 16oz deli containers",
    "and some deli containers",
    "how many per case for the 16oz deli?",
]


def _render_cart(cart: Cart) -> str:
    if cart.is_empty():
        return "    (empty)"
    lines = []
    for supplier, items in cart.by_supplier.items():
        lines.append(f"    [{supplier}]")
        for item in items:
            lines.append(f"      - {item.quantity} x {item.product_name} ({item.sku})")
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    if not settings.google_api_key:
        sys.exit("GOOGLE_API_KEY is not set — add it to .env first.")

    configure_logging()
    log = logging.getLogger(__name__)
    tracing = configure_tracing(settings)
    log.info("model=%s embeddings=%s", settings.gemini_model, settings.embedding_model)
    log.info(tracing_status_line(settings, tracing))
    log.info("building agent (downloads model + builds FAISS index on first run)…")
    agent = build_agent()

    cart = Cart()
    for i, turn in enumerate(TURNS):
        trace = TraceContext(surface="smoke", metadata={"turn_index": i}) if tracing else None
        result = handle_turn(turn, cart, agent, trace=trace)
        cart = result.cart  # persist across turns, like the UI's gr.State
        print(f"USER: {turn}")
        print(f"AGENT: {result.reply}")
        print("CART:")
        print(_render_cart(cart))
        print("-" * 70)


if __name__ == "__main__":
    main()