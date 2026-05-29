"""gate — the confidence/clarification centerpiece (spec §3, thesis: ask only when unsure).

Decides clarify vs draft from the turn's line items:
  * any item below the confidence threshold  -> clarify
  * any blocking flag present (even at high confidence) -> clarify
  * otherwise -> draft
Pure and deterministic — the probabilistic part (was the confidence right?) is
covered by the eval set, not here.
"""

from src.app.graph.gates import gate
from src.app.graph.policies import BLOCKING_FLAGS, CONFIDENCE_THRESHOLD
from src.domain.models import Flag, LineItem


def test_clean_high_confidence_item_drafts():
    assert gate([LineItem(sku="DELI-16", confidence=0.95)]) == "draft"


def test_low_confidence_item_clarifies():
    assert gate([LineItem(sku="DELI-16", confidence=CONFIDENCE_THRESHOLD - 0.1)]) == "clarify"


def test_confidence_exactly_at_threshold_is_confident_enough_to_draft():
    assert gate([LineItem(sku="DELI-16", confidence=CONFIDENCE_THRESHOLD)]) == "draft"


def test_blocking_flag_clarifies_even_at_high_confidence():
    item = LineItem(sku="DELI-16", confidence=0.99, flags=[Flag.NEEDS_LIDS])
    assert gate([item]) == "clarify"


def test_every_blocking_flag_forces_clarify():
    for flag in BLOCKING_FLAGS:
        assert gate([LineItem(sku="X", confidence=0.99, flags=[flag])]) == "clarify"


def test_informational_flag_does_not_block_drafting():
    # ROUNDED_TO_CASE_PACK is informational only — not a blocking flag.
    assert Flag.ROUNDED_TO_CASE_PACK not in BLOCKING_FLAGS
    item = LineItem(sku="DELI-16", confidence=0.99, flags=[Flag.ROUNDED_TO_CASE_PACK])
    assert gate([item]) == "draft"


def test_any_single_low_confidence_item_triggers_clarify():
    items = [LineItem(sku="A", confidence=0.99), LineItem(sku="B", confidence=0.2)]
    assert gate(items) == "clarify"


def test_no_line_items_drafts():
    assert gate([]) == "draft"