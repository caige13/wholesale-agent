"""Eval judges — the deterministic scoring behind the two metrics.

The eval's LLM runs are probabilistic, but *grading* them isn't: given the
agent's final cart and the expected row, the scores are pure functions. So they're
unit-tested here (keyless), and the eval runner just applies them.
"""

from types import SimpleNamespace

from evals.judges import JudgeVerdict, answer_faithfulness, clarification_correct, extraction_score
from src.domain.cart import Cart
from src.domain.models import LineItem

S = "acme-foodservice"


class FakeJudge:
    """Stands in for the GPT-4o judge used via with_structured_output(...).invoke()."""

    def __init__(self, verdict: JudgeVerdict):
        self._verdict = verdict

    def with_structured_output(self, _schema):
        return SimpleNamespace(invoke=lambda _prompt: self._verdict)


def _cart(*pairs) -> Cart:
    lines = [LineItem(sku=sku, supplier=S, quantity=qty) for sku, qty in pairs]
    return Cart(by_supplier={S: lines}) if lines else Cart()


def test_extraction_scores_one_for_an_exact_match():
    expected = {"skus": ["STRAW-WRAP"], "quantities": [2]}
    assert extraction_score(expected, _cart(("STRAW-WRAP", 2))) == 1.0


def test_extraction_scores_one_when_both_expected_and_cart_are_empty():
    # the ambiguous case: nothing should be committed
    assert extraction_score({"skus": [], "quantities": []}, Cart()) == 1.0


def test_extraction_gives_partial_credit_for_a_partial_match():
    expected = {"skus": ["FORK-PLAS", "FOIL-ROLL"], "quantities": [2, 1]}
    score = extraction_score(expected, _cart(("FORK-PLAS", 2)))  # got 1 of 2
    assert 0.0 < score < 1.0


def test_extraction_scores_zero_for_the_wrong_quantity():
    expected = {"skus": ["DELI-16"], "quantities": [3]}
    assert extraction_score(expected, _cart(("DELI-16", 2))) == 0.0


def test_extraction_uses_cart_unchanged_for_a_question_turn():
    before = _cart(("DELI-16", 2))
    expected = {"skus": [], "quantities": [], "cart_unchanged": True}
    assert extraction_score(expected, before, cart_before=before) == 1.0
    # a question turn that mutated the cart is wrong
    assert extraction_score(expected, Cart(), cart_before=before) == 0.0


def test_clarification_is_correct_only_when_asking_matches_expectation():
    assert clarification_correct(expected=True, asked=True) is True
    assert clarification_correct(expected=False, asked=False) is True
    assert clarification_correct(expected=True, asked=False) is False
    assert clarification_correct(expected=False, asked=True) is False


def test_answer_faithfulness_returns_the_judge_verdict():
    contex = "16oz Deli Container (16oz, 500 per case)"
    yes = FakeJudge(JudgeVerdict(faithful=True))
    no = FakeJudge(JudgeVerdict(faithful=False))
    assert answer_faithfulness("how many per case?", contex, "500 per case", yes) is True
    assert answer_faithfulness("how many per case?", contex, "a dozen", no) is False