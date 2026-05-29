"""Graph policy constants — the tunable knobs the gate's decision depends on.

Kept here (app layer) rather than in the domain so the *policy* of when to ask
for clarification lives with the graph, while the domain only defines what the
flags mean.
"""

from src.domain.models import Flag

# An item resolved below this confidence triggers a clarifying question.
# Items at or above it are treated as confident.
CONFIDENCE_THRESHOLD = 0.6

# Flags that force a clarifying question even when confidence is high.
# ROUNDED_TO_CASE_PACK is deliberately excluded — it's informational, not blocking.
BLOCKING_FLAGS = frozenset(
    {
        Flag.AMBIGUOUS_SIZE,
        Flag.OUT_OF_STOCK,
        Flag.NEEDS_LIDS,
        Flag.BELOW_MINIMUM,
    }
)