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


def test_drafts_when_all_items_are_high_confidence():
    assert gate([LineItem(sku="DELI-16", confidence=0.95)]) == "draft"


def test_clarifies_when_an_item_is_below_the_confidence_threshold():
    assert gate([LineItem(sku="DELI-16", confidence=CONFIDENCE_THRESHOLD - 0.1)]) == "clarify"


def test_drafts_when_confidence_is_exactly_at_the_threshold():
    assert gate([LineItem(sku="DELI-16", confidence=CONFIDENCE_THRESHOLD)]) == "draft"


def test_clarifies_on_a_blocking_flag_even_at_high_confidence():
    item = LineItem(sku="DELI-16", confidence=0.99, flags=[Flag.NEEDS_LIDS])
    assert gate([item]) == "clarify"


def test_every_blocking_flag_forces_clarification():
    for flag in BLOCKING_FLAGS:
        assert gate([LineItem(sku="X", confidence=0.99, flags=[flag])]) == "clarify"


def test_drafts_when_only_an_informational_flag_is_present():
    # ROUNDED_TO_CASE_PACK is informational only — not a blocking flag.
    assert Flag.ROUNDED_TO_CASE_PACK not in BLOCKING_FLAGS
    item = LineItem(sku="DELI-16", confidence=0.99, flags=[Flag.ROUNDED_TO_CASE_PACK])
    assert gate([item]) == "draft"


def test_clarifies_when_any_single_item_is_low_confidence():
    items = [LineItem(sku="A", confidence=0.99), LineItem(sku="B", confidence=0.2)]
    assert gate(items) == "clarify"


def test_drafts_when_there_are_no_line_items():
    assert gate([]) == "draft"


def test_unscored_line_defaults_to_clarify():
    # confidence defaults to 0.0, so a line resolution never scored is treated as
    # low-confidence and clarifies — the fail-safe that underpins "ask when unsure".
    assert gate([LineItem(sku="DELI-16")]) == "clarify"