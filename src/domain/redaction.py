"""redact_normalize — the graph's front-door guardrail (pure, regex-based, no LLM).

Runs as the first node so phone/email accidentally typed into an order message
are stripped before anything downstream — the LLM, the tools, the LangSmith
trace — ever sees them. Also normalizes spaced size units ("16 oz" -> "16oz") so
later parsing/resolution see a consistent form, while leaving order quantities
("3 cases", "1200") untouched.

``pii_found`` records only the *type* that fired ("phone"/"email"), never the
value, so even our own state and traces don't hold the PII.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Common US phone shapes: optional +1, (555), and -/./space separators, plus a
# bare 10-digit run. Bounded by (?<!\d)/(?!\d) so it can't bite into a longer
# digit string, and 10 digits means order quantities (<=4 digits) never match.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"
)

# Size units only. Order matters: longest alternatives first so "ounces" wins
# over "oz" and "lb" over "l". Case/pack are deliberately absent — they're
# quantities, not sizes, and must survive ("3 cases").
_UNIT_CANONICAL = {
    "ounces": "oz", "ounce": "oz", "oz": "oz",
    "pounds": "lb", "pound": "lb", "lbs": "lb", "lb": "lb",
    "gallons": "gal", "gallon": "gal", "gal": "gal",
    "quarts": "qt", "quart": "qt", "qt": "qt",
    "pints": "pt", "pint": "pt", "pt": "pt",
    "liters": "l", "liter": "l", "ml": "ml", "l": "l",
    "grams": "g", "gram": "g", "kg": "kg", "g": "g",
    "count": "ct", "ct": "ct",
}
_UNITS_ALTERNATION = "|".join(sorted(_UNIT_CANONICAL, key=len, reverse=True))
_UNIT_RE = re.compile(rf"(\d+)\s+({_UNITS_ALTERNATION})\b", re.IGNORECASE)


class RedactionResult(BaseModel):
    """Output of the front-door guardrail: the cleaned message and which PII types fired."""

    clean_message: str
    pii_found: list[str] = Field(default_factory=list)


def redact_normalize(text: str) -> RedactionResult:
    pii_found: list[str] = []

    def _redact(label: str, placeholder: str):
        def _sub(_match: re.Match) -> str:
            pii_found.append(label)
            return placeholder

        return _sub

    cleaned = _EMAIL_RE.sub(_redact("email", "[REDACTED_EMAIL]"), text)
    cleaned = _PHONE_RE.sub(_redact("phone", "[REDACTED_PHONE]"), cleaned)
    cleaned = _UNIT_RE.sub(lambda m: m.group(1) + _UNIT_CANONICAL[m.group(2).lower()], cleaned)

    return RedactionResult(clean_message=cleaned, pii_found=pii_found)