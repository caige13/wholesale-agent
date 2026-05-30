"""gate — the clarification decision. The thesis of the agent: ask only when unsure.

A pure function over the turn's line items, used as the graph's conditional edge
(clarify vs draft). Keeping it deterministic means the "did we ask at the right
time?" behavior is unit-tested here and measured by the clarification eval metric.
"""

from __future__ import annotations

from typing import Literal

from src.domain.models import LineItem
from src.domain.policies import BLOCKING_FLAGS, CONFIDENCE_THRESHOLD

Decision = Literal["clarify", "draft"]


def gate(line_items: list[LineItem]) -> Decision:
    """Return "clarify" if any item is low-confidence or carries a blocking flag,
    otherwise "draft".
    """
    for item in line_items:
        if item.confidence < CONFIDENCE_THRESHOLD:
            return "clarify"
        if any(flag in BLOCKING_FLAGS for flag in item.flags):
            return "clarify"
    return "draft"