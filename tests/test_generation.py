"""Offline tests for the LLM backends.

Local backends are exercised against httpx.MockTransport, so the whole suite
still runs with no model, no server and no API key.
"""

import json

import httpx
import pytest

from generation import llm as llm_mod
from generation.llm import (
    AnthropicBackend,
    LLMResponse,
    OllamaBackend,
    OpenAICompatibleBackend,
    available_backends,
    get_llm,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- cost accounting ----------------------------------------------------


def test_hosted_cost_uses_the_pricing_table():
    response = LLMResponse(text="x", model="claude-opus-5", backend="anthropic",
                           input_tokens=1_000_000, output_tokens=1_000_000)
    assert response.cost_usd == pytest.approx(30.00)


def test_local_model_costs_nothing_but_still_reports_tokens():
    response = LLMResponse(text="x", model="llama3.1:8b", backend="ollama",
                           input_tokens=900, output_tokens=120)
    assert response.cost_usd == 0.0
    assert (response.input_tokens, response.output_tokens) == (900, 120)


# --- Ollama -------------------------------------------------------------


def _ndjson(*events) -> bytes:
    return b"".join((json.dumps(e) + "\n").encode() for e in events)


def test_ollama_parses_response_and_token_counts():
    def handler(request):
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.0
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        return httpx.Response(200, json={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "  this is a test.  "},
            "prompt_eval_count": 42,
            "eval_count": 7,
            "done_reason": "stop",
        })

    backend = OllamaBackend(stream=False, client=_client(handler))
    response = backend.complete("hi", system="be terse")

    assert response.text == "this is a test."
    assert response.backend == "ollama"
    assert (response.input_tokens, response.output_tokens) == (42, 7)
    assert response.cost_usd == 0.0
    assert response.latency_s >= 0


def test_ollama_omits_system_message_when_not_given():
    def handler(request):
        assert [m["role"] for m in json.loads(request.content)["messages"]] == ["user"]
        return httpx.Response(200, json={"message": {"content": "ok"}})

    assert OllamaBackend(stream=False, client=_client(handler)).complete("hi").text == "ok"


# --- streaming (the default) --------------------------------------------


def test_ollama_streams_by_default_and_reassembles_tokens():
    """Non-streaming returns no bytes until generation ends, so the read
    timeout has to cover model load + prompt eval + every token at once."""
    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=_ndjson(
            {"model": "llama3.1:8b", "message": {"content": "this "}},
            {"model": "llama3.1:8b", "message": {"content": "is a "}},
            {"model": "llama3.1:8b", "message": {"content": "test."},
             "done": True, "done_reason": "stop",
             "prompt_eval_count": 1300, "eval_count": 12},
        ))

    response = OllamaBackend(client=_client(handler)).complete("hi")
    assert response.text == "this is a test."
    assert (response.input_tokens, response.output_tokens) == (1300, 12)
    assert response.model == "llama3.1:8b"


def test_ollama_stream_skips_unparseable_lines():
    def handler(request):
        return httpx.Response(200, content=(
            b'{"message":{"content":"ok"}}\n' b"not json\n"
            b'{"message":{"content":"!"},"done":true,"eval_count":2}\n'))

    assert OllamaBackend(client=_client(handler)).complete("x").text == "ok!"


def test_ollama_keeps_the_model_resident():
    """Ollama unloads an idle model; reloading 4.7GB dominates the next query."""
    def handler(request):
        assert json.loads(request.content)["keep_alive"] == "30m"
        return httpx.Response(200, content=_ndjson({"message": {"content": "ok"},
                                                    "done": True}))

    OllamaBackend(client=_client(handler)).complete("x")


def test_ollama_read_timeout_becomes_an_actionable_error():
    from generation.llm import LLMTimeout

    def handler(request):
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(LLMTimeout, match="produced no output"):
        OllamaBackend(client=_client(handler)).complete("x")


def test_ollama_connect_error_names_the_likely_cause():
    from generation.llm import LLMTimeout

    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMTimeout, match="ollama serve"):
        OllamaBackend(client=_client(handler)).complete("x")


def test_warmup_loads_the_model_without_generating():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"message": {"content": ""}})

    OllamaBackend(client=_client(handler)).warmup()
    assert calls[0]["messages"] == []
    assert calls[0]["keep_alive"] == "30m"


def test_warmup_survives_an_unreachable_server():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert OllamaBackend(client=_client(handler)).warmup() >= 0


def test_ollama_available_reflects_server_state():
    up = OllamaBackend(client=_client(lambda r: httpx.Response(200, json={"models": []})))
    assert up.available() is True

    def boom(request):
        raise httpx.ConnectError("refused")

    assert OllamaBackend(client=_client(boom)).available() is False


def test_ollama_lists_installed_models():
    handler = lambda r: httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    assert OllamaBackend(client=_client(handler)).list_models() == ["llama3.1:8b"]


def test_ollama_raises_on_http_error():
    handler = lambda r: httpx.Response(404, json={"error": "model not found"})
    with pytest.raises(httpx.HTTPStatusError):
        OllamaBackend(client=_client(handler)).complete("hi")


# --- OpenAI-compatible local servers ------------------------------------


def test_openai_compatible_parses_chat_completion():
    def handler(request):
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json={
            "model": "qwen2.5-7b",
            "choices": [{"message": {"content": "this is a test."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 5},
        })

    response = OpenAICompatibleBackend(client=_client(handler)).complete("hi")
    assert response.text == "this is a test."
    assert (response.input_tokens, response.output_tokens) == (30, 5)
    assert response.backend == "openai"


def test_openai_compatible_survives_missing_usage_block():
    """llama.cpp builds sometimes omit `usage`; the eval harness must not crash."""
    handler = lambda r: httpx.Response(200, json={
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
    })
    response = OpenAICompatibleBackend(client=_client(handler)).complete("x")
    assert (response.input_tokens, response.output_tokens) == (0, 0)
    assert response.text == "hi"


def test_openai_compatible_survives_empty_choices():
    handler = lambda r: httpx.Response(200, json={"choices": []})
    assert OpenAICompatibleBackend(client=_client(handler)).complete("x").text == ""


def test_openai_compatible_sends_bearer_header():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer not-needed"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    OpenAICompatibleBackend(client=_client(handler)).complete("x")


# --- backend selection --------------------------------------------------


def test_placeholder_api_key_is_not_treated_as_real(monkeypatch):
    """The repo's .env ships `your_api_key_here`; routing to Anthropic on that
    produces a confusing 401 instead of falling through to a local model."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_api_key_here")
    assert llm_mod._real_anthropic_key() is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc123")
    assert llm_mod._real_anthropic_key() == "sk-ant-abc123"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert llm_mod._real_anthropic_key() is None


def test_auto_prefers_anthropic_when_the_key_is_real(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc123")
    monkeypatch.delenv("RAG_LLM_BACKEND", raising=False)
    assert get_llm().name == "anthropic"


def test_auto_falls_back_to_a_running_local_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_api_key_here")
    monkeypatch.delenv("RAG_LLM_BACKEND", raising=False)
    monkeypatch.setattr(OllamaBackend, "available", lambda self: True)
    assert get_llm().name == "ollama"


def test_auto_raises_with_actionable_guidance_when_nothing_is_available(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RAG_LLM_BACKEND", raising=False)
    monkeypatch.setattr(OllamaBackend, "available", lambda self: False)
    monkeypatch.setattr(OpenAICompatibleBackend, "available", lambda self: False)

    with pytest.raises(RuntimeError, match="ollama serve"):
        get_llm()


def test_env_var_selects_the_backend(monkeypatch):
    monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
    assert get_llm().name == "ollama"


def test_explicit_backend_overrides_env_and_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc123")
    monkeypatch.setenv("RAG_LLM_BACKEND", "anthropic")
    assert get_llm("ollama").name == "ollama"


def test_explicit_model_is_passed_through():
    assert get_llm("ollama", model="qwen2.5:7b").model == "qwen2.5:7b"
    assert get_llm("anthropic", model="claude-sonnet-5").model == "claude-sonnet-5"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_llm("gpt4all")


def test_available_backends_probes_all_three(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(OllamaBackend, "available", lambda self: True)
    monkeypatch.setattr(OpenAICompatibleBackend, "available", lambda self: False)
    assert available_backends() == {
        "anthropic": False, "groq": False, "ollama": True, "openai": False, "stub": True,
    }


def test_anthropic_backend_defaults_to_opus_5():
    assert AnthropicBackend().model == "claude-opus-5"


# --- stub backend -------------------------------------------------------


def test_stub_quotes_first_evidence_and_cites_it():
    from generation.llm import StubBackend

    response = StubBackend().complete("Evidence:\n[1] Roth IRAs are post-tax.\n\nQuestion: q")
    assert response.text == "Roth IRAs are post-tax. [1]"
    assert response.backend == "stub"
    assert response.cost_usd == 0.0


def test_stub_abstains_when_there_is_no_evidence():
    from generation.llm import StubBackend

    assert StubBackend().complete("Evidence:\n(none)\n\nQuestion: q").text == \
        "INSUFFICIENT_EVIDENCE"


def test_auto_never_selects_the_stub(monkeypatch):
    """A fake answer must never reach results without being asked for."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RAG_LLM_BACKEND", raising=False)
    monkeypatch.setattr(OllamaBackend, "available", lambda self: False)
    monkeypatch.setattr(OpenAICompatibleBackend, "available", lambda self: False)
    with pytest.raises(RuntimeError):
        get_llm()
    assert get_llm("stub").name == "stub"
