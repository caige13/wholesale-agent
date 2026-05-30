"""FAISS catalog repository — real embeddings + FAISS (integration).

Verifies the RAG retrieval seam end-to-end: semantic search returns scored
domain candidates, ordered, in range, and that "salsa cups" actually lands on
the portion-cup SKU. Skipped automatically when the agent deps aren't installed.
"""

import pytest

pytest.importorskip("faiss")
pytest.importorskip("langchain_huggingface")

from src.bootstrap import build_catalog_repository  # noqa: E402
from src.domain.models import ResolutionCandidate  # noqa: E402
from src.ports import CatalogRepository  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def faiss_repo() -> CatalogRepository:
    # Builds the embedder + FAISS index once for the module (model load is slow).
    return build_catalog_repository()


def test_satisfies_the_catalog_repository_port(faiss_repo):
    assert isinstance(faiss_repo, CatalogRepository)


def test_get_and_all_delegate_to_the_loaded_catalog(faiss_repo):
    assert faiss_repo.get("DELI-16").product_name == "16oz Deli Container"
    assert len(faiss_repo.all()) >= 15


def test_find_candidates_returns_scored_domain_candidates(faiss_repo):
    results = faiss_repo.find_candidates("deli containers", k=5)
    assert results
    assert all(isinstance(c, ResolutionCandidate) for c in results)
    assert all(0.0 <= c.score <= 1.0 for c in results)


def test_find_candidates_orders_results_by_descending_score(faiss_repo):
    scores = [c.score for c in faiss_repo.find_candidates("fresh limes", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_semantically_matches_salsa_cups_to_the_portion_cup_sku(faiss_repo):
    skus = [c.item.sku for c in faiss_repo.find_candidates("salsa cups", k=5)]
    assert "PCUP-2" in skus


def test_respects_the_k_limit(faiss_repo):
    assert len(faiss_repo.find_candidates("containers", k=3)) <= 3