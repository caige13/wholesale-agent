"""Eval runner — score the agent over the dataset on four metrics.

Three are deterministic (extraction correctness, clarification behavior, order
submission); the fourth (answer faithfulness, question turns only) is judged by a
different model (OpenAI) so it never grades its own work. Every agent/judge call
is auto-traced to LangSmith when tracing is enabled, so the runs are visible there
alongside the printed scores.

Run:  uv run python -m evals.run_eval   (needs GOOGLE_API_KEY; OPENAI_API_KEY for the judge)

Known gaps (documented future work, not bugs):
  * reorder_usual — item-memory isn't populated yet, so "the usual" can't resolve;
    until that lands the row expects the agent to ask for clarification (correct
    behavior given there's no reorder logic), not to reconstruct a remembered cart.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals._data import cart_from, format_context, load_rows  # noqa: E402
from evals.judges import (  # noqa: E402
    answer_faithfulness,
    clarification_correct,
    escalation_correct,
    extraction_score,
    submission_correct,
)
from src.app.graph.agent import LangGraphOrderAgent  # noqa: E402
from src.app.graph.graph import build_graph  # noqa: E402
from src.bootstrap import (  # noqa: E402
    build_catalog_repository,
    build_chat_model,
    build_escalation_gateway,
    build_judge_model,
    build_supplier_gateway,
)
from src.config import get_settings  # noqa: E402
from src.observability import (  # noqa: E402
    TraceContext,
    configure_logging,
    configure_tracing,
    tracing_status_line,
)

_log = logging.getLogger(__name__)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    settings = get_settings()
    if not settings.google_api_key:
        sys.exit("GOOGLE_API_KEY is not set — add it to .env first.")

    configure_logging()
    tracing = configure_tracing(settings)
    _log.info(tracing_status_line(settings, tracing))

    rows = load_rows()
    catalog = build_catalog_repository()
    agent = LangGraphOrderAgent(
        build_graph(
            build_chat_model(),
            catalog,
            build_supplier_gateway(),
            escalation=build_escalation_gateway(),
        )
    )
    judge = build_judge_model() if settings.openai_api_key else None
    if judge is None:
        _log.warning("no OPENAI_API_KEY — skipping the answer-faithfulness judge")

    extraction: list[float] = []
    clarification: list[bool] = []
    submission: list[bool] = []
    escalation: list[bool] = []
    faithfulness: list[bool] = []

    print(f"{'id':<22}{'extract':>8}{'clarify':>9}{'submit':>8}{'escal':>7}{'answer':>9}")
    print("-" * 63)
    for row in rows:
        before = cart_from(row.get("starting_cart"))
        # Optional per-row history lets a row exercise a multi-turn follow-up
        # (e.g. answering a pending add-on offer); absent ⇒ None, single-turn.
        # Tag the run with the row id so each eval turn is findable in LangSmith.
        trace = (
            TraceContext(surface="eval", metadata={"eval_row_id": row["id"]}) if tracing else None
        )
        result = agent.run(row["input_message"], before, row.get("history"), trace=trace)
        expected = row["expected"]

        ext = extraction_score(expected, result.draft_cart, before)
        clr = clarification_correct(expected["expects_clarification"], bool(result.clarifications))
        # Draft vs. place: only an explicit checkout should yield a supplier
        # confirmation; absent the key, the row expects a running draft (no submit).
        sub = submission_correct(expected.get("submitted", False), bool(result.confirmation))
        # Escalation: a handoff ticket should appear exactly on the escalation rows.
        esc = escalation_correct(expected.get("expects_escalation", False), bool(result.handoff))
        extraction.append(ext)
        clarification.append(clr)
        submission.append(sub)
        escalation.append(esc)

        ans = "-"
        if result.answer and judge is not None:
            context = format_context(catalog.find_candidates(row["input_message"]))
            try:
                faithful = answer_faithfulness(row["input_message"], context, result.answer, judge)
                faithfulness.append(faithful)
                ans = "ok" if faithful else "FAIL"
            except Exception as exc:  # noqa: BLE001 — a dead judge shouldn't sink the run
                ans = "err"
                judge = None  # stop hammering an unavailable judge
                _log.warning("answer judge unavailable: %s — skipping it", type(exc).__name__)

        print(
            f"{row['id']:<22}{ext:>8.2f}{('ok' if clr else 'FAIL'):>9}"
            f"{('ok' if sub else 'FAIL'):>8}{('ok' if esc else 'FAIL'):>7}{ans:>9}"
        )
        if ext < 1.0 or not clr or not sub or not esc:  # show what happened, to diagnose the miss
            pairs = sorted((li.sku, li.quantity) for li in result.draft_cart.all_lines())
            print(
                f"    cart={pairs}  asked={result.clarifications}  "
                f"submitted={bool(result.confirmation)}  escalated={bool(result.handoff)}"
            )

    print("-" * 63)
    print(f"extraction correctness : {_mean(extraction):.0%}")
    print(f"clarification behavior : {_mean([float(c) for c in clarification]):.0%}")
    print(f"order submission       : {_mean([float(s) for s in submission]):.0%}")
    print(f"escalation behavior    : {_mean([float(e) for e in escalation]):.0%}")
    if faithfulness:
        print(f"answer faithfulness    : {_mean([float(f) for f in faithfulness]):.0%}")


if __name__ == "__main__":
    main()