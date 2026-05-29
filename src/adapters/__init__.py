"""Adapters — concrete implementations of ports. The only place vendor/tech
specifics (JSON files, FAISS, provider SDKs) appear.
"""

from src.adapters.json_catalog_repository import JsonCatalogRepository

__all__ = ["JsonCatalogRepository"]