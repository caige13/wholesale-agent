"""Companion add-ons — the deterministic sizing math (no LLM, no catalog).

``companion_case_count`` sizes a companion order so its unit count covers the
parent's without going under — the math the LLM must *not* do. Which companions
are still *under-covered* (and so still worth offering) is a catalog-dependent
question, so it lives in the graph layer (``app.graph.nodes._undercovered_companions``).
"""

from __future__ import annotations

import math


def companion_case_count(parent_units: int, companion_case_pack: int) -> int:
    """Cases of a companion needed to cover ``parent_units`` without going under.

    e.g. 3 cases of 32oz deli (480/case = 1440 units) paired with a lid packed
    500/case → ``ceil(1440 / 500) = 3`` cases (1500 lids ≥ 1440). Always ≥ 1 so a
    parent with an unknown/zero unit count still gets a single case offered.
    """
    if companion_case_pack <= 0:
        return 1
    return max(1, math.ceil(parent_units / companion_case_pack))