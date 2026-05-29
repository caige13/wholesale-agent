r"""Guards the eval dataset's shape (spec §9) so the deferred eval runner can
rely on it. This is a *unit* test of a fixture file — it does not hit an LLM.
"""

import json
from pathlib import Path

import pytest

_DATASET = Path(__file__).resolve().parents[2] / "evals" / "datasets" / "order_desk.jsonl"

REQUIRED_EXPECTED_KEYS = {"skus", "suppliers", "quantities", "expects_clarification"}
OPTIONAL_EXPECTED_KEYS = {"cart_unchanged", "pii_redacted"}


def _load_rows() -> list[dict]:
    lines = [ln for ln in _DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def test_has_enough_rows():
    rows = _load_rows()
    assert 10 <= len(rows) <= 20  # spec §9: 10-15 rows (a little headroom)


def test_row_ids_are_unique():
    rows = _load_rows()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("row", _load_rows(), ids=lambda r: r["id"])
def test_each_row_matches_the_spec_schema(row: dict):
    assert isinstance(row["input_message"], str) and row["input_message"]
    expected = row["expected"]
    assert REQUIRED_EXPECTED_KEYS <= expected.keys()
    assert expected.keys() <= REQUIRED_EXPECTED_KEYS | OPTIONAL_EXPECTED_KEYS
    # skus / suppliers / quantities are parallel lists.
    assert len(expected["skus"]) == len(expected["suppliers"]) == len(expected["quantities"])
    assert isinstance(expected["expects_clarification"], bool)
    if "cart_before" in row:
        assert isinstance(row["cart_before"], dict)


def test_includes_every_required_eval_case():
    ids = {r["id"] for r in _load_rows()}
    # The spec §9 "Required cases" — every one must have a row.
    required = {
        "clean_single",
        "ambiguous_size",
        "alias_salsa_cups",
        "missing_lids",
        "out_of_stock",
        "reorder_usual",
        "modify_quantity",
        "remove_limes",
        "interleaved_question",
        "pii_phone",
    }
    assert required <= ids


def test_the_interleaved_question_case_marks_the_cart_unchanged():
    row = next(r for r in _load_rows() if r["id"] == "interleaved_question")
    assert row["expected"].get("cart_unchanged") is True