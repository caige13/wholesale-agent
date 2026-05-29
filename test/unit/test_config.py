"""Settings loader behavior — defaults, env overrides, blank-normalization, caching.

config.py has real logic (env reads + defaults + lru_cache), so it's a genuine
test target (unlike the type/Protocol declarations, which we don't unit-test).
"""

import pytest

from src.config import get_settings

_ENV_VARS = [
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "EMBEDDING_MODEL",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
]


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    # get_settings() is cached; clear around each test so env changes take effect.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_apply_when_env_absent(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    s = get_settings()
    assert s.gemini_model == "gemini-2.5-flash"
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.langsmith_project == "ai-order-desk"
    assert s.google_api_key is None
    assert s.langsmith_tracing is False


def test_env_overrides_win(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "key-123")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    s = get_settings()
    assert s.gemini_model == "gemini-test"
    assert s.google_api_key == "key-123"
    assert s.langsmith_tracing is True


def test_blank_api_key_normalized_to_none(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    assert get_settings().google_api_key is None


def test_settings_are_cached(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert get_settings() is get_settings()