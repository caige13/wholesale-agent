r"""JsonCatalogRepository — loads kb/catalog.json into domain CatalogItems."""

import pytest

from src.adapters import JsonCatalogRepository
from src.domain.models import CatalogItem
from src.ports import CatalogRepository


def test_adapter_satisfies_catalog_repository_port(catalog_repo: JsonCatalogRepository):
    # Guards the hexagonal seam: the JSON adapter must honor the port contract.
    assert isinstance(catalog_repo, CatalogRepository)


def test_loads_all_catalog_items(catalog_repo: JsonCatalogRepository):
    items = catalog_repo.all()
    assert len(items) >= 15  # spec §8: a mini KB of ~15-20 SKUs
    assert all(isinstance(i, CatalogItem) for i in items)


def test_get_returns_item_by_sku(catalog_repo: JsonCatalogRepository):
    item = catalog_repo.get("DELI-16")
    assert item is not None
    assert item.product_name == "16oz Deli Container"
    assert item.supplier == "acme-foodservice"


def test_get_unknown_sku_returns_none(catalog_repo: JsonCatalogRepository):
    assert catalog_repo.get("NOPE-999") is None


def test_skus_are_unique(catalog_repo: JsonCatalogRepository):
    skus = [i.sku for i in catalog_repo.all()]
    assert len(skus) == len(set(skus))


def test_catalog_carries_no_price_or_stock(catalog_repo: JsonCatalogRepository):
    # Price/stock are dynamic supplier-API data, never in the RAG corpus.
    item = catalog_repo.get("DELI-16")
    assert not hasattr(item, "price")
    assert not hasattr(item, "in_stock")


def test_ambiguous_phrase_has_multiple_size_variants(catalog_repo: JsonCatalogRepository):
    # "deli containers" must be genuinely ambiguous for the clarification eval.
    deli = [i for i in catalog_repo.all() if "deli container" in i.product_name.lower()]
    assert len({i.unit_size for i in deli}) >= 2


def test_salsa_cups_alias_resolves_to_single_sku(catalog_repo: JsonCatalogRepository):
    matches = [i for i in catalog_repo.all() if "salsa cups" in i.aliases]
    assert [i.sku for i in matches] == ["PCUP-2"]


def test_find_candidates_is_deferred(catalog_repo: JsonCatalogRepository):
    with pytest.raises(NotImplementedError):
        catalog_repo.find_candidates("salsa cups")