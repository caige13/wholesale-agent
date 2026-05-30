"""Eval runner — score the agent over the dataset on three metrics.

Two are deterministic (extraction correctness, clarification behavior); the third
(answer faithfulness, question turns only) is judged by a different model (OpenAI)
so it never grades its own work. Every agent/judge call is auto-traced to
LangSmith when tracing is enabled, so the runs are visible there alongside the
printed scores.

Run:  uv run python -m evals.run_eval   (needs GOOGLE_API_KEY; OPENAI_API_KEY for the judge)

Known gaps the eval will surface (documented future work, not bugs):
  * out_of_stock  — inventory/SupplierGateway is deferred, so it won't flag/ask.
  * reorder_usual — item-memory isn't populated yet, so "the usual" won't resolve.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.judges import answer_faithfulness, clarification_correct, extraction_score  # noqa: E402
from src.app.graph.agent import LangGraphOrderAgent  # noqa: E402
from src.app.graph.graph import build_graph  # noqa: E402
from src.bootstrap import (  # noqa: E402
    build_catalog_repository,
    build_chat_model,
    build_judge_model,
)
from src.config import get_settings  # noqa: E402
from src.domain.cart import Cart  # noqa: E402
from src.domain.models import LineItem  # noqa: E402

_DATASET = Path(__file__).resolve().parent / "datasets" / "order_desk.jsonl"


def _load_rows() -> list[dict]:
    lines = [ln for ln in _DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _cart_from(raw: dict | None) -> Cart:
    if not raw:
        return Cart()
    return Cart(by_supplier={s: [LineItem(**li) for li in items] for s, items in raw.items()})


def _format_context(candidates) -> str:
    if not candidates:
        return "(no matching catalog entries)"
    return "\n".join(
        f"- {c.item.product_name} ({c.item.unit_size}, {c.item.case_pack} per case)"
        for c in candidates
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    settings = get_settings()
    if not settings.google_api_key:
        sys.exit("GOOGLE_API_KEY is not set — add it to .env first.")

    rows = _load_rows()
    catalog = build_catalog_repository()
    agent = LangGraphOrderAgent(build_graph(build_chat_model(), catalog))
    judge = build_judge_model() if settings.openai_api_key else None
    if judge is None:
        print("(no OPENAI_API_KEY — skipping the answer-faithfulness judge)\n")

    extraction: list[float] = []
    clarification: list[bool] = []
    faithfulness: list[bool] = []

    print(f"{'id':<22}{'extract':>8}{'clarify':>9}{'answer':>9}")
    print("-" * 48)
    for row in rows:
        before = _cart_from(row.get("cart_before"))
        result = agent.run(row["input_message"], before)
        expected = row["expected"]

        ext = extraction_score(expected, result.draft_cart, before)
        clr = clarification_correct(expected["expects_clarification"], bool(result.clarifications))
        extraction.append(ext)
        clarification.append(clr)

        ans = "-"
        if result.answer and judge is not None:
            context = _format_context(catalog.find_candidates(row["input_message"]))
            faithful = answer_faithfulness(row["input_message"], context, result.answer, judge)
            faithfulness.append(faithful)
            ans = "ok" if faithful else "FAIL"

        print(f"{row['id']:<22}{ext:>8.2f}{('ok' if clr else 'FAIL'):>9}{ans:>9}")

    print("-" * 48)
    print(f"extraction correctness : {_mean(extraction):.0%}")
    print(f"clarification behavior : {_mean([float(c) for c in clarification]):.0%}")
    if faithfulness:
        print(f"answer faithfulness    : {_mean([float(f) for f in faithfulness]):.0%}")


if __name__ == "__main__":
    main()