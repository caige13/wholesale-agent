"""Escalation gateway port — the human-handoff seam.

When the agent can't (or shouldn't) handle a turn itself — the user asks for a
person, or for something outside the order desk (returns, billing, disputes) — it
hands off to a human. This Protocol is the boundary to whatever support system
takes over (a ticketing API, a call queue); ``MockEscalationGateway`` stands in for
it, exactly as ``MockSupplierGateway`` stands in for the supplier API.

Like the other gateways the agent never sees the wire format: an adapter translates
it into a domain ``Handoff`` behind this boundary.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.domain.models import Handoff


@runtime_checkable
class EscalationGateway(Protocol):
    def create_handoff(self, reason: str, summary: str) -> Handoff:
        """Open a human-handoff ticket for this turn and return its acknowledgement.

        ``reason`` is a short tag for *why* the handoff happened (the user's request);
        ``summary`` is the conversational context the human picks the thread up from.
        """
        ...