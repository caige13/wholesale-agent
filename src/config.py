"""Runtime settings — the single env-backed config surface.

This is also the **composition seam for the generic AI plumbing**: later slices
build the LangChain chat model / embeddings / vector store from these values
(e.g. ``init_chat_model(settings.gemini_model)``), so the provider is swappable
without touching any node. Nothing here is imported by the deterministic core or
its tests — `make test` needs no env and no API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Frozen snapshot of environment configuration."""

    google_api_key: str | None
    gemini_model: str
    # Client-side throttle (requests/min) so we stay under the model's quota.
    # Default is safe for the Gemini free tier (5/min); raise it on a paid tier.
    gemini_rpm: int
    embedding_model: str
    langsmith_tracing: bool
    langsmith_api_key: str | None
    langsmith_project: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once. Cached so repeated calls are cheap and consistent."""
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_rpm=int(os.getenv("GEMINI_RPM", "5")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "").lower() == "true",
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY") or None,
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "ai-order-desk"),
    )