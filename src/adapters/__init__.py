"""Adapters — concrete implementations of ports. The only place vendor/tech
specifics (JSON files, FAISS, provider SDKs) appear.
"""

from src.adapters.json_catalog_repository import JsonCatalogRepository
from src.adapters.mock_supplier_gateway import MockSupplierGateway

__all__ = ["JsonCatalogRepository", "MockSupplierGateway"]