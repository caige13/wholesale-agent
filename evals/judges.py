"""Deterministic scoring for the two eval metrics (spec §9), kept separate.

These are pure functions over the agent's output and the expected row, so they're
unit-tested directly. The eval runner applies them to each (probabilistic) agent
run; quality of the run is the LLM's job, grading it is not.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.domain.cart import Cart


def extraction_score(
    expected: dict,
    final_cart: Cart,
    cart_before: Cart | None = None,
) -> float:
    """How well the final cart matches what the turn should have produced.

    A question turn (``cart_unchanged``) scores on the cart being untouched.
    Otherwise it's the overlap (Jaccard) of (sku, quantity) pairs — partial
    credit when some lines are right.
    """
    if expected.get("cart_unchanged"):
        return 1.0 if _pairs(final_cart) == _pairs(cart_before or Cart()) else 0.0

    expected_pairs = set(zip(expected["skus"], expected["quantities"], strict=True))
    actual_pairs = _pairs(final_cart)
    if not expected_pairs and not actual_pairs:
        return 1.0
    union = expected_pairs | actual_pairs
    return len(expected_pairs & actual_pairs) / len(union)


def clarification_correct(expected: bool, asked: bool) -> bool:
    """The thesis metric: did the agent ask for clarification exactly when it should?"""
    return expected == asked


class JudgeVerdict(BaseModel):
    """An LLM judge's structured verdict on an open-ended answer."""

    faithful: bool
    reasoning: str = ""


_FAITHFULNESS_PROMPT = (
    "You are grading an assistant's answer to a restaurant's product question. "
    "Judge ONLY whether the answer is faithful to — i.e. supported by and "
    "consistent with — the catalog context, and not fabricated. Answer faithful "
    "= true/false.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
)


def answer_faithfulness(question: str, context: str, answer: str, judge) -> bool:
    """LLM-as-judge for the open-ended RAG answer (reference-free faithfulness).

    The judge model is injected (GPT-4o at runtime, a fake in tests) — a different
    model from the agent (Gemini), so it doesn't grade its own work. Used only for
    free-text answers; the structured metrics are graded deterministically above.
    """
    verdict = judge.with_structured_output(JudgeVerdict).invoke(
        _FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    )
    return verdict.faithful


def _pairs(cart: Cart) -> set[tuple[str | None, int | None]]:
    return {(line.sku, line.quantity) for line in cart.all_lines()}