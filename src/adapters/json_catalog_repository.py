"""JSON-backed catalog repository — loads the static product catalog from disk.

Implements the ``get``/``all`` half of ``CatalogRepository``. Semantic
``find_candidates`` is implemented by ``FaissCatalogRepository``
(``src/adapters/faiss_catalog_repository.py``), a FAISS-backed adapter that
embeds ``product_name + aliases + category`` and returns scored
``ResolutionCandidate``s (LangChain ``Document`` → domain object, the
anti-corruption boundary).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.domain.models import CatalogItem, ResolutionCandidate

# Repo-root-relative default: src/adapters/ -> repo root -> kb/catalog.json
_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "kb" / "catalog.json"


class JsonCatalogRepository:
    """Loads ``kb/catalog.json`` once and serves catalog items by SKU."""

    def __init__(self, catalog_path: str | Path = _DEFAULT_CATALOG_PATH) -> None:
        path = Path(catalog_path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        self._by_sku: dict[str, CatalogItem] = {}
        for row in rows:
            item = CatalogItem(**row)
            if item.sku in self._by_sku:
                raise ValueError(f"duplicate SKU in catalog: {item.sku}")
            self._by_sku[item.sku] = item

    def get(self, sku: str) -> CatalogItem | None:
        return self._by_sku.get(sku)

    def all(self) -> list[CatalogItem]:
        return list(self._by_sku.values())

    def find_candidates(self, query: str, k: int = 5) -> list[ResolutionCandidate]:
        raise NotImplementedError(
            "Semantic retrieval lands in the RAG slice (FAISS over "
            "product_name + aliases + category). resolve_skus is unit-tested "
            "with candidates passed in directly."
        )