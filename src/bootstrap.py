"""Composition root — the one place that knows every concrete adapter.

Everywhere else depends only on ports. This wires the real (agent-group) stack:
the local embedding model and the FAISS-backed catalog. Tests don't import this
— they wire their own fakes/fixtures. Only the runtime / integration tests do.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.adapters import JsonCatalogRepository
from src.adapters.faiss_catalog_repository import FaissCatalogRepository
from src.config import get_settings
from src.ports import CatalogRepository


def build_embeddings() -> Embeddings:
    """The local, keyless sentence-transformers embedder (normalized for cosine)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=get_settings().embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_catalog_repository(embeddings: Embeddings | None = None) -> CatalogRepository:
    """Load kb/catalog.json and index it for semantic retrieval."""
    items = JsonCatalogRepository().all()
    return FaissCatalogRepository(items, embeddings or build_embeddings())


def build_chat_model():
    """The Gemini chat model (needs GOOGLE_API_KEY). temperature=0 for stability."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


def build_agent(item_memory: dict[str, str] | None = None):
    """Wire the full order-desk agent: Gemini + FAISS catalog behind the graph."""
    from src.app.graph.agent import LangGraphOrderAgent
    from src.app.graph.graph import build_graph

    graph = build_graph(build_chat_model(), build_catalog_repository(), item_memory)
    return LangGraphOrderAgent(graph)