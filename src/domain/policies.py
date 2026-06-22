"""Business-policy constants — the tunable knobs behind the agent's decisions.

Domain-layer on purpose: both ``resolve_skus`` (domain) and ``gate`` (app) read
from one place, and app may depend on domain but never the reverse. Co-locating
these makes the key coupling explicit — ``AMBIGUOUS_CONFIDENCE`` must stay below
``CONFIDENCE_THRESHOLD``, or "ask only when unsure" silently breaks (the
resolve_skus tests assert exactly that boundary).
"""

from src.domain.models import Flag

# --- Clarification gate -----------------------------------------------------
# An item resolved below this confidence triggers a clarifying question; at or
# above it the item is treated as confident.
CONFIDENCE_THRESHOLD = 0.6

# Flags that force a clarifying question even at high confidence.
# ROUNDED_TO_CASE_PACK is excluded — it's informational, not blocking.
BLOCKING_FLAGS = frozenset(
    {
        Flag.AMBIGUOUS_SIZE,
        Flag.OUT_OF_STOCK,
        Flag.EXCEEDS_STOCK,
        Flag.NEEDS_COMPANION,
        Flag.BELOW_MINIMUM,
        Flag.MISSING_QUANTITY,
    }
)

# A subset of BLOCKING_FLAGS: these also keep the line OUT of the cart until the
# customer resolves it — the order is wrong as stated (ambiguous size, un/over-
# fillable, below minimum, or missing a quantity), so don't bank it; ask first.
# NEEDS_COMPANION is deliberately excluded: the parent line is valid and lands while
# we merely upsell the add-on (apply_node owns this; the gate still clarifies on all
# blocking flags).
HOLD_FROM_CART_FLAGS = BLOCKING_FLAGS - frozenset({Flag.NEEDS_COMPANION})

# --- SKU resolution ---------------------------------------------------------
MIN_SCORE = 0.5  # a top candidate below this is too weak to commit to
MARGIN = 0.15  # the top must beat the runner-up by at least this to be unambiguous
MEMORY_CONFIDENCE = 0.95  # item-memory ("the usual") resolves with near-certainty
ALIAS_CONFIDENCE = 0.9  # an exact alias / product-name match
AMBIGUOUS_CONFIDENCE = 0.4  # below CONFIDENCE_THRESHOLD → forces a clarifying question