"""Configuration: env parsing, provider defaults, and the fail-fast key checks.

Every knob in this project is an env var, so the coercion helpers and the
per-provider defaults are load-bearing. These tests reload `config` with a
stubbed `load_dotenv`, so they assert the *shipped* defaults rather than
whatever happens to be in a local .env file.
"""

import importlib

import pytest

import config

CONFIG_ENV_VARS = [
    "LLM_PROVIDER", "EMBED_PROVIDER", "GOOGLE_API_KEY", "OPENAI_API_KEY",
    "CHAT_MODEL", "EMBEDDING_MODEL", "EMBED_DEVICE", "EMBED_NORMALIZE",
    "LLAMA_N_CTX", "LLAMA_N_THREADS", "LLAMA_N_BATCH", "LLAMA_MAX_TOKENS",
    "CITATION_MODE", "CITE_THRESHOLD", "CITE_MIN_CHARS", "TEMPERATURE",
    "DATA_DIR", "INDEX_DIR", "CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_CHUNK_CHARS",
    "TOP_K", "FETCH_K", "MMR_LAMBDA", "USE_HYBRID", "WEIGHT_DENSE",
    "WEIGHT_SPARSE", "EMBED_BATCH_SIZE", "EMBED_MAX_RETRIES", "EMBED_RPM",
    "EMBED_CACHE", "EMBED_CACHE_DIR", "HOST", "PORT", "MAX_HISTORY_TURNS",
    "ALLOWED_ORIGINS",
]


@pytest.fixture(autouse=True)
def _restore_real_config():
    """Autouse so it tears down *after* monkeypatch has undone the env."""
    yield
    importlib.reload(config)


@pytest.fixture
def fresh_config(monkeypatch):
    """Reload config from a known-empty environment, ignoring any .env on disk."""

    def _load(**env):
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        for name in CONFIG_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        return importlib.reload(config)

    return _load


# -- coercion helpers ----------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on", " on "])
def test_bool_accepts_the_documented_truthy_spellings(value, monkeypatch):
    monkeypatch.setenv("SOME_FLAG", value)
    assert config._bool("SOME_FLAG", False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "nonsense"])
def test_bool_treats_everything_else_as_false(value, monkeypatch):
    monkeypatch.setenv("SOME_FLAG", value)
    assert config._bool("SOME_FLAG", True) is False


def test_bool_falls_back_to_its_default(monkeypatch):
    monkeypatch.delenv("SOME_FLAG", raising=False)
    assert config._bool("SOME_FLAG", True) is True
    assert config._bool("SOME_FLAG", False) is False


def test_int_and_float_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("SOME_INT", "42")
    monkeypatch.setenv("SOME_FLOAT", "0.25")
    assert config._int("SOME_INT", 1) == 42
    assert config._float("SOME_FLOAT", 1.0) == 0.25


def test_int_and_float_fall_back_to_defaults(monkeypatch):
    monkeypatch.delenv("MISSING", raising=False)
    assert config._int("MISSING", 7) == 7
    assert config._float("MISSING", 1.5) == 1.5


# -- provider defaults ---------------------------------------------------------


def test_local_provider_defaults_to_qwen_and_bge(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="local")
    assert cfg.CHAT_MODEL == "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    assert cfg.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"


def test_openai_provider_defaults(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="openai")
    assert cfg.CHAT_MODEL == "gpt-4o-mini"
    assert cfg.EMBEDDING_MODEL == "text-embedding-3-small"


def test_google_provider_defaults(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="google")
    assert cfg.CHAT_MODEL == "gemini-flash-latest"
    assert cfg.EMBEDDING_MODEL == "models/gemini-embedding-001"


def test_the_provider_name_is_normalised(fresh_config):
    assert fresh_config(LLM_PROVIDER="  LOCAL  ").PROVIDER == "local"


def test_embeddings_follow_the_chat_provider_by_default(fresh_config):
    assert fresh_config(LLM_PROVIDER="openai").EMBED_PROVIDER == "openai"


def test_embeddings_can_use_a_different_provider(fresh_config):
    """Local embeddings plus hosted chat dodges the quota wall that actually bites."""
    cfg = fresh_config(LLM_PROVIDER="google", EMBED_PROVIDER="local")
    assert cfg.PROVIDER == "google" and cfg.EMBED_PROVIDER == "local"
    assert cfg.CHAT_MODEL == "gemini-flash-latest"
    assert cfg.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"


def test_explicit_models_override_the_provider_defaults(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="openai", CHAT_MODEL="gpt-4o", EMBEDDING_MODEL="custom")
    assert cfg.CHAT_MODEL == "gpt-4o" and cfg.EMBEDDING_MODEL == "custom"


# -- provider-dependent defaults -----------------------------------------------


def test_local_defaults_to_computed_citations(fresh_config):
    """Models this small do not emit [n] markers, so auto is the honest default."""
    assert fresh_config(LLM_PROVIDER="local").CITATION_MODE == "auto"


def test_hosted_providers_trust_the_model_to_cite(fresh_config):
    assert fresh_config(LLM_PROVIDER="google").CITATION_MODE == "model"


def test_local_embedding_is_unthrottled_and_batched_large(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="local")
    assert cfg.EMBED_RPM == 0
    assert cfg.EMBED_BATCH_SIZE == 64


def test_hosted_embedding_is_throttled_and_batched_small(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="google")
    assert cfg.EMBED_RPM == 90
    assert cfg.EMBED_BATCH_SIZE == 20


def test_mmr_ships_disabled(fresh_config):
    """Measured as harmful on this corpus; lambda 1.0 makes MMR a no-op."""
    assert fresh_config(LLM_PROVIDER="local").MMR_LAMBDA == 1.0


def test_retrieval_defaults(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="local")
    assert (cfg.TOP_K, cfg.FETCH_K, cfg.USE_HYBRID) == (5, 20, True)
    assert cfg.HYBRID_WEIGHTS == (0.6, 0.4)
    assert (cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP, cfg.MIN_CHUNK_CHARS) == (1000, 150, 80)


def test_allowed_origins_splits_and_strips(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="local", ALLOWED_ORIGINS="https://a.com, https://b.com ,")
    assert cfg.ALLOWED_ORIGINS == ["https://a.com", "https://b.com"]


# -- key checks ----------------------------------------------------------------


def test_local_needs_no_key_for_chat_or_embedding(fresh_config):
    """This is the property that makes a clean checkout work with no signup."""
    cfg = fresh_config(LLM_PROVIDER="local")
    cfg.require_api_key()
    cfg.require_embed_key()


def test_google_without_a_key_fails_with_an_actionable_message(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="google")
    with pytest.raises(SystemExit, match="GOOGLE_API_KEY"):
        cfg.require_api_key()


def test_openai_without_a_key_fails_with_an_actionable_message(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="openai")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        cfg.require_api_key()


def test_a_present_key_satisfies_the_check(fresh_config):
    fresh_config(LLM_PROVIDER="openai", OPENAI_API_KEY="sk-test").require_api_key()
    fresh_config(LLM_PROVIDER="google", GOOGLE_API_KEY="g-test").require_api_key()


def test_embed_key_check_follows_the_embed_provider(fresh_config):
    """Chat on a keyless provider must not excuse a keyless hosted embedder."""
    cfg = fresh_config(LLM_PROVIDER="local", EMBED_PROVIDER="google")
    cfg.require_api_key()  # chat is local: fine
    with pytest.raises(SystemExit, match="EMBED_PROVIDER=google"):
        cfg.require_embed_key()


def test_local_embeddings_need_no_key_even_with_hosted_chat(fresh_config):
    fresh_config(LLM_PROVIDER="google", EMBED_PROVIDER="local").require_embed_key()


def test_active_api_key_picks_the_right_one(fresh_config):
    cfg = fresh_config(LLM_PROVIDER="openai", OPENAI_API_KEY="sk-x", GOOGLE_API_KEY="g-y")
    assert cfg.active_api_key() == "sk-x"
    cfg = fresh_config(LLM_PROVIDER="google", OPENAI_API_KEY="sk-x", GOOGLE_API_KEY="g-y")
    assert cfg.active_api_key() == "g-y"


def test_summary_names_both_providers_and_the_knobs_that_move(fresh_config):
    summary = fresh_config(LLM_PROVIDER="local").summary()
    for fragment in ("chat=local", "embed=local", "chunk=1000/150", "top_k=5", "hybrid=True"):
        assert fragment in summary
