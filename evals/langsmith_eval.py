"""LangSmith-native eval — push the dataset and run ``evaluate()`` in-platform.

The local ``run_eval`` runner prints scores to the terminal; this is its versioned
sibling. It upserts ``evals/datasets/order_desk.jsonl`` into a LangSmith Dataset and
runs ``langsmith.evaluate()`` so every run is a comparable, drill-down-able
experiment in the UI (model-vs-model, per-row, over time).

Crucially it does **not** re-implement scoring: the evaluators wrap the very same
deterministic functions and judge in ``evals/judges.py`` that the local runner uses,
so the two paths can never drift.

Run:  uv run python -m evals.langsmith_eval
Needs: LANGSMITH_API_KEY + LANGSMITH_TRACING=true, GOOGLE_API_KEY (agent),
       OPENAI_API_KEY (the faithfulness judge — optional; skipped if absent).
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langsmith import Client, evaluate  # noqa: E402

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
from src.domain.cart import Cart  # noqa: E402
from src.observability import configure_logging, configure_tracing  # noqa: E402

_log = logging.getLogger(__name__)

DATASET_NAME = "order-desk"
# Stable namespace so a row's example id is deterministic across re-runs — that's
# what makes the dataset upsert idempotent (we only create ids we haven't yet).
_NS = uuid.UUID("00000000-0000-0000-0000-0000000de5c0")


def _example_id(row_id: str) -> uuid.UUID:
    return uuid.uuid5(_NS, row_id)


def _example_payload(row: dict) -> dict:
    """The create/update payload for one row — keyed by a deterministic example id so
    re-runs target the same example. (``ExampleCreate``/``ExampleUpdate`` share this shape.)

    Optional inputs (``history``, ``starting_cart``) are omitted when absent rather than
    sent as null — the target/evaluators read them with ``.get()``, and omitting keeps the
    stored example byte-stable so the diff-sync sees an untouched row as *unchanged*.
    """
    inputs = {"input_message": row["input_message"]}
    if row.get("history") is not None:
        inputs["history"] = row["history"]
    if row.get("starting_cart") is not None:
        inputs["starting_cart"] = row["starting_cart"]
    return {
        "id": _example_id(row["id"]),
        "inputs": inputs,
        "outputs": row["expected"],
        "metadata": {"row_id": row["id"]},
    }


def upsert_dataset(client: Client, rows: list[dict]) -> None:
    """Make the LangSmith dataset mirror the JSONL: create missing rows, update the rest.

    The JSONL is the source of truth, so this *syncs* rather than only-appends — an edit
    (a renamed input key, a corrected ``expected``) propagates on the next run instead of
    silently leaving stale examples behind. Examples are matched by a deterministic id.
    """
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            DATASET_NAME, description="AI Order Desk behavioral eval (mirrors order_desk.jsonl)."
        )

    existing = {ex.id: ex for ex in client.list_examples(dataset_id=dataset.id)}
    to_create, to_update = [], []
    for row in rows:
        payload = _example_payload(row)
        current = existing.get(payload["id"])
        if current is None:
            to_create.append(payload)
        elif current.inputs != payload["inputs"] or current.outputs != payload["outputs"]:
            # Only push rows whose content actually changed, so a no-op run doesn't
            # churn a new example version on every example.
            to_update.append(payload)
    if to_create:
        client.create_examples(dataset_id=dataset.id, examples=to_create)
    if to_update:
        client.update_examples(dataset_id=dataset.id, updates=to_update)
    _log.info(
        "dataset '%s': %d created, %d updated, %d unchanged",
        DATASET_NAME,
        len(to_create),
        len(to_update),
        len(rows) - len(to_create) - len(to_update),
    )


def make_target(agent):
    """A ``(inputs) -> outputs`` callable that runs one turn through the agent.

    Returns the cart (serialized so an evaluator can rebuild it) plus the decision
    signals the deterministic metrics need. No explicit trace — ``evaluate`` already
    traces each target call as the experiment run.
    """

    def target(inputs: dict) -> dict:
        before = cart_from(inputs.get("starting_cart"))
        result = agent.run(inputs["input_message"], before, inputs.get("history"))
        return {
            "cart": result.draft_cart.model_dump(mode="json"),
            "clarifications": result.clarifications,
            "answer": result.answer,
            "submitted": bool(result.confirmation),
            "escalated": bool(result.handoff),
        }

    return target


# --- evaluators: thin adapters over evals/judges.py (one source of truth) ------


def extraction_evaluator(run, example) -> dict:
    final_cart = Cart.model_validate(run.outputs["cart"])
    before = cart_from(example.inputs.get("starting_cart"))
    return {"key": "extraction", "score": extraction_score(example.outputs, final_cart, before)}


def clarification_evaluator(run, example) -> dict:
    asked = bool(run.outputs["clarifications"])
    correct = clarification_correct(example.outputs["expects_clarification"], asked)
    return {"key": "clarification", "score": correct}


def submission_evaluator(run, example) -> dict:
    correct = submission_correct(example.outputs.get("submitted", False), run.outputs["submitted"])
    return {"key": "submission", "score": correct}


def escalation_evaluator(run, example) -> dict:
    correct = escalation_correct(
        example.outputs.get("expects_escalation", False), run.outputs["escalated"]
    )
    return {"key": "escalation", "score": correct}


def make_faithfulness_evaluator(catalog, judge):
    """Faithfulness is question-only and cross-model; scored on the answer turns."""

    def faithfulness_evaluator(run, example) -> dict:
        answer = run.outputs.get("answer")
        if not answer:  # not a question turn — record a comment, no score
            return {"key": "faithfulness", "comment": "n/a — not a question turn"}
        context = format_context(catalog.find_candidates(example.inputs["input_message"]))
        faithful = answer_faithfulness(example.inputs["input_message"], context, answer, judge)
        return {"key": "faithfulness", "score": faithful}

    return faithfulness_evaluator


def main() -> None:
    settings = get_settings()
    if not settings.langsmith_api_key:
        sys.exit("LANGSMITH_API_KEY is not set — needed to push the dataset + experiment.")
    if not settings.google_api_key:
        sys.exit("GOOGLE_API_KEY is not set — add it to .env first.")
    configure_logging()
    configure_tracing(settings)

    client = Client()
    rows = load_rows()
    upsert_dataset(client, rows)

    # Build the catalog once and share it between the agent and the judge's context
    # lookup, so we embed the catalog a single time.
    catalog = build_catalog_repository()
    agent = LangGraphOrderAgent(
        build_graph(
            build_chat_model(),
            catalog,
            build_supplier_gateway(),
            escalation=build_escalation_gateway(),
        )
    )
    evaluators = [
        extraction_evaluator,
        clarification_evaluator,
        submission_evaluator,
        escalation_evaluator,
    ]
    if settings.openai_api_key:
        evaluators.append(make_faithfulness_evaluator(catalog, build_judge_model()))
    else:
        _log.warning("no OPENAI_API_KEY — skipping the answer-faithfulness evaluator")

    results = evaluate(
        make_target(agent),
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=settings.gemini_model,
        metadata={"model": settings.gemini_model, "judge": settings.judge_model},
        client=client,
        # Sequential: the agent's chat model is rate-limited (Gemini free tier).
        max_concurrency=1,
    )
    # The SDK prints the experiment URL; surface the name too for the terminal log.
    print(f"experiment: {getattr(results, 'experiment_name', DATASET_NAME)}")


if __name__ == "__main__":
    main()