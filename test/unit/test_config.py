"""Settings loader behavior — defaults, env overrides, blank-normalization, caching.

config.py has real logic (env reads + defaults + lru_cache), so it's a genuine
test target (unlike the type/Protocol declarations, which we don't unit-test).
"""

import pytest

from src.config import get_settings

_ENV_VARS = [
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_RPM",
    "OPENAI_API_KEY",
    "JUDGE_MODEL",
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


def test_applies_defaults_when_env_vars_are_absent(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    s = get_settings()
    assert s.gemini_model == "gemini-2.5-flash"
    assert s.gemini_rpm == 5  # safe under the Gemini free-tier 5 req/min quota
    assert s.judge_model == "gpt-4o"  # the (cross-model) eval judge
    assert s.openai_api_key is None
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.langsmith_project == "ai-order-desk"
    assert s.google_api_key is None
    assert s.langsmith_tracing is False


def test_env_vars_override_the_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "key-123")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    s = get_settings()
    assert s.gemini_model == "gemini-test"
    assert s.google_api_key == "key-123"
    assert s.langsmith_tracing is True


def test_reads_gemini_rpm_as_an_int_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_RPM", "60")
    assert get_settings().gemini_rpm == 60


def test_normalizes_a_blank_api_key_to_none(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    assert get_settings().google_api_key is None


def test_returns_a_cached_instance_on_repeated_calls(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert get_settings() is get_settings()