"""Mock escalation gateway — stands in for the human-handoff system.

Deterministic and keyless, like ``MockSupplierGateway``: it mints a stable ticket
id from the handoff's reason+summary (no clock, no randomness), so the same handoff
always yields the same id, and returns a fixed support callback number + ETA. A real
adapter would POST to a ticketing/queue API behind the same ``EscalationGateway``
port; nothing else in the system changes.
"""

from __future__ import annotations

import hashlib

from src.domain.models import Handoff

_DEFAULT_CALLBACK = "1-800-555-0123"
_DEFAULT_ETA_MINUTES = 15


class MockEscalationGateway:
    """Opens a deterministic handoff ticket; fakes the support-system round-trip."""

    def __init__(
        self,
        callback_number: str = _DEFAULT_CALLBACK,
        eta_minutes: int = _DEFAULT_ETA_MINUTES,
    ) -> None:
        self._callback_number = callback_number
        self._eta_minutes = eta_minutes

    def create_handoff(self, reason: str, summary: str) -> Handoff:
        return Handoff(
            ticket_id=self._ticket_id(reason, summary),
            reason=reason,
            callback_number=self._callback_number,
            eta_minutes=self._eta_minutes,
        )

    def _ticket_id(self, reason: str, summary: str) -> str:
        """A stable id from (reason, summary) — deterministic so the same handoff
        always mints the same ticket (no clock / randomness)."""
        digest = hashlib.sha1(f"{reason}|{summary}".encode()).hexdigest()[:8].upper()
        return f"SUP-{digest}"