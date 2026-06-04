"""Observability — structured logging plus the LangSmith run-label schema.

Both edge concerns live here so the deterministic core and its keyless tests never
import LangSmith:

* ``configure_logging`` installs a single structured log format for the entry points.
* ``configure_tracing`` applies the ``Settings`` snapshot to the process env so the
  LangChain tracer activates — previously these settings were read but never applied,
  so tracing worked only because ``.env`` happened to use the canonical var names.
* ``TraceContext`` defines the **run-label schema** in one place, so every run is
  labeled the same way and is sliceable in the LangSmith UI.

### What's labeled, and where it comes from

Labels are set **at run creation** via the invoke config — conflict-free,
boundary-only (the nodes stay pure), and zero extra HTTP:

    run name          order_desk_{surface}
    tags              ["order-desk", "surface:{surface}"]
    metadata.surface  "ui" | "eval" | "smoke"
    metadata.*        surface-specific: eval_row_id / history_len / turn_index

The **outcome** of the turn (intent, status, clarifications, confirmation) is *not*
re-attached as metadata: it's already captured as the run's **outputs** (they're keys
in the graph's final state), so a reviewer sees it on the trace for free. Attaching it
again post-hoc would mean patching an already-finalized run — which races the
background tracer (a 409) and blocks on the network — so we don't.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import Settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
# Our own top-level logger namespaces — only these emit at INFO; everything else
# (httpx, langsmith, gradio, sentence-transformers …) stays at WARNING so neither
# production nor the eval console is drowned in framework chatter.
_APP_LOGGERS = ("src", "evals", "scripts")


def configure_logging(level: int = logging.INFO) -> None:
    """Install one structured log format for an entry point (idempotent).

    The root logger — and thus every third-party library — is held at WARNING; only
    this app's loggers emit at ``level``. ``basicConfig`` is a no-op once the root has
    a handler, so calling this from each entry point is safe and consistent.
    """
    logging.basicConfig(level=logging.WARNING, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(level)


def configure_tracing(settings: Settings) -> bool:
    """Apply LangSmith settings to the process env and return whether tracing is on.

    Idempotent. When disabled, force ``LANGSMITH_TRACING=false`` so a stray env var
    can't silently turn it back on. The LangChain tracer reads these on each run.
    """
    import os

    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return True


def tracing_status_line(settings: Settings, active: bool) -> str:
    """A one-line, log-friendly summary of the tracing state for entry points."""
    if not active:
        return "LangSmith tracing: off"
    return f"LangSmith tracing: on -> project '{settings.langsmith_project}'"


@dataclass(frozen=True)
class TraceContext:
    """Per-turn labels attached to one ``graph.invoke`` for LangSmith.

    ``surface`` is where the turn came from ("ui" | "eval" | "smoke"); it names the
    run and tags it so the UI can filter. ``metadata`` is the surface-specific slice of
    the schema (e.g. ``eval_row_id``, ``history_len``). Callers pass a ``TraceContext``
    only when tracing is on; otherwise they pass ``None`` and the run is unlabeled.
    """

    surface: str = "app"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_runnable_config(self) -> dict[str, Any]:
        """The ``RunnableConfig`` slice for ``graph.invoke`` — run name + tags + metadata."""
        return {
            "run_name": f"order_desk_{self.surface}",
            "tags": ["order-desk", f"surface:{self.surface}"],
            "metadata": {"surface": self.surface, **self.metadata},
        }