"""Project-root pytest fixtures.

Session-scoped fixtures for things that are expensive or shared across the unit,
integration, and (later) eval suites. The real (FAISS/embedder) adapters live in
the deferred `agent` group, so for now this only wires the keyless JSON catalog.
"""

import pytest

from src.adapters import JsonCatalogRepository


@pytest.fixture(scope="session")
def catalog_repo() -> JsonCatalogRepository:
    """The real JSON-backed catalog (kb/catalog.json), loaded once per session."""
    return JsonCatalogRepository()