"""resolve_skus — choose a SKU for a line from scored retriever candidates.

Deterministic arbitration, not matching: the semantic work happens upstream in
the embedding retriever (which produces the scored candidates); this function
just decides among them. Precedence:
  1. item-memory  — a learned phrase->SKU mapping (reorder / "the usual") wins outright
  2. exact alias / name match among the candidates (high-confidence fast path)
  3. a clearly-best candidate (top score high enough AND well ahead of the runner-up)
  4. otherwise leave it unresolved at low confidence — the gate then asks.

Confidence is the gate's input, so getting it right here is what makes the agent
ask only when it's genuinely unsure. The thresholds live in ``src.domain.policies``.
"""

from __future__ import annotations

from src.domain.models import CatalogItem, LineItem, ResolutionCandidate
from src.domain.policies import (
    ALIAS_CONFIDENCE,
    AMBIGUOUS_CONFIDENCE,
    MARGIN,
    MEMORY_CONFIDENCE,
    MIN_SCORE,
)


def resolve_skus(
    item: LineItem,
    candidates: list[ResolutionCandidate],
    item_memory: dict[str, str] | None = None,
) -> LineItem:
    memory = item_memory or {}
    phrase = item.raw_text.strip().lower()

    # 1. Item-memory wins outright (reorder / learned alias).
    if phrase in memory:
        target = _candidate_for_sku(candidates, memory[phrase])
        if target is not None:
            return _resolved(item, target, MEMORY_CONFIDENCE)
        return item.model_copy(update={"sku": memory[phrase], "confidence": MEMORY_CONFIDENCE})

    if not candidates:
        return item.model_copy(update={"confidence": 0.0})

    # 2. Exact alias / product-name match — high-confidence fast path.
    exact = _exact_match(candidates, phrase)
    if exact is not None:
        return _resolved(item, exact, ALIAS_CONFIDENCE)

    # 2b. Size disambiguation: when a size token in the phrase ("16oz") matches
    #     exactly one candidate's unit_size, it resolves a same-family cluster.
    sized = _size_match(candidates, phrase)
    if sized is not None:
        return _resolved(item, sized, ALIAS_CONFIDENCE)

    # 3. A clearly-best candidate: strong enough and well ahead of the runner-up.
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = ranked[0]
    runner_up = ranked[1].score if len(ranked) > 1 else 0.0
    if top.score >= MIN_SCORE and (top.score - runner_up) >= MARGIN:
        return _resolved(item, top.item, top.score)

    # 4. Too close or too weak → unresolved, low confidence (the gate will clarify),
    #    carrying the close candidates so the clarifying question can offer them.
    options = [c.item.product_name for c in ranked[:3]]
    return item.model_copy(
        update={"sku": None, "confidence": AMBIGUOUS_CONFIDENCE, "options": options}
    )


def _resolved(item: LineItem, catalog_item: CatalogItem, confidence: float) -> LineItem:
    return item.model_copy(
        update={
            "sku": catalog_item.sku,
            "product_name": catalog_item.product_name,
            "supplier": catalog_item.supplier,
            "unit": catalog_item.unit_size,
            "confidence": confidence,
        }
    )


def _candidate_for_sku(candidates: list[ResolutionCandidate], sku: str) -> CatalogItem | None:
    return next((c.item for c in candidates if c.item.sku == sku), None)


def _size_match(candidates: list[ResolutionCandidate], phrase: str) -> CatalogItem | None:
    """Resolve when a phrase token equals exactly one candidate's unit_size.

    Token (not substring) match so "16oz" doesn't spuriously match "6oz". Returns
    None unless exactly one candidate matches, so a genuine ambiguity stays open.
    """
    tokens = phrase.split()
    matches = [c.item for c in candidates if c.item.unit_size.lower() in tokens]
    return matches[0] if len(matches) == 1 else None


def _exact_match(candidates: list[ResolutionCandidate], phrase: str) -> CatalogItem | None:
    """Match when a candidate's product name or an alias is contained in the phrase.

    Containment (not just equality) lets a qualifier/plural still match — e.g.
    "16oz deli containers" contains the alias "16oz deli container" — which
    disambiguates same-family items the embedding scores cluster together. Bare
    "deli containers" still matches no specific alias, so it stays ambiguous.
    """
    for c in candidates:
        names = [c.item.product_name.lower(), *(a.lower() for a in c.item.aliases)]
        if any(name in phrase for name in names):
            return c.item
    return None