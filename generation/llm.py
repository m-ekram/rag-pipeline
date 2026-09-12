"""Pluggable LLM backends for grounded generation.

The rest of the pipeline is already fully local: bge-small embeds on-device,
Qdrant runs in Docker, BM25 is in-process. Generation is the *only* stage that
ever needed a hosted API, so it is the only stage that needs this abstraction.

Three backends, one interface:

  anthropic  Claude via the official SDK. Costs money, needs ANTHROPIC_API_KEY.
  ollama     Ollama's native /api/chat. Free, offline, reports real token counts.
  openai     Any OpenAI-compatible /v1/chat/completions server — llama.cpp's
             llama-server, LM Studio, vLLM, text-generation-webui, and Ollama's
             own compatibility endpoint. "openai" names the *wire protocol*
             here, not the vendor; no OpenAI service is contacted.

Every backend returns the same `LLMResponse`, so the eval harness can report
tokens, latency and cost per query without knowing which one ran. Local
backends report cost 0.0 — the number is still real, it is just zero.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import httpx

from generation.prompts import estimate_tokens

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Local backends are free by construction.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_GROQ_URL = os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1")
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_LOCAL_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:8080/v1")
DEFAULT_LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "local-model")
# Local models on CPU are slow; generous timeout prevents mid-answer cutoffs.
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "600"))
# Connecting to a local server is instant or never; only *reading* is slow.
# One scalar timeout conflates them, so a dead server waits the full read
# budget before failing.
DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("LLM_CONNECT_TIMEOUT", "10"))
# Ollama unloads an idle model and reloads it on the next call (~4.7GB for an
# 8B). Keeping it resident is the difference between a 2-minute first token and
# a 2-second one on later questions.
DEFAULT_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
# Starting context window for Ollama requests, grown on demand (see
# OllamaBackend._options). 4096 fits the local preset's prompt with headroom.
DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
MAX_NUM_CTX = 16384
# Installed Ollama models to prefer, in order: small enough to answer in
# seconds on a laptop CPU, large enough to follow the grounding rules.
PREFERRED_OLLAMA_MODELS = ("qwen2.5:3b", "llama3.2:3b", "qwen2.5:1.5b", "llama3.2:1b", "llama3.1:8b")


def preferred_ollama_order(names: list[str]) -> list[str]:
    """Installed model names with the preferred ones first, the rest as given."""
    def rank(name: str) -> int:
        for position, prefix in enumerate(PREFERRED_OLLAMA_MODELS):
            if name.startswith(prefix):
                return position
        return len(PREFERRED_OLLAMA_MODELS)

    return sorted(names, key=rank)  # stable, so smallest-first order survives within a rank


def _timeout(read: float) -> "httpx.Timeout":
    return httpx.Timeout(read, connect=DEFAULT_CONNECT_TIMEOUT, write=read, pool=read)


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        rates = PRICING.get(self.model)
        if not rates:
            return 0.0  # local model, or an unpriced/unknown hosted one
        return (self.input_tokens * rates[0] + self.output_tokens * rates[1]) / 1_000_000


class LLMTimeout(RuntimeError):
    """The local server accepted the request but produced nothing in time."""


class LLMRefusal(RuntimeError):
    """The model declined to answer for safety reasons (not a transport error)."""


class LLMBackend(Protocol):
    name: str
    model: str

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 stream_callback: Optional[Any] = None) -> LLMResponse: ...

    def available(self) -> bool: ...


# --- Anthropic ----------------------------------------------------------


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, *,
                 api_key: Optional[str] = None,
                 client: Optional[Any] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def available(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("sk-ant-"))

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 stream_callback: Optional[Any] = None) -> LLMResponse:
        # temperature is accepted for interface parity but not sent: current
        # Claude models reject sampling parameters.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        started = time.perf_counter()
        message = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - started

        # A safety refusal returns HTTP 200 with stop_reason "refusal" and
        # usually no text block. Left unchecked it would surface as an empty
        # ungrounded answer instead of a distinguishable failure.
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMRefusal(
                f"{self.model} declined the request"
                + (f" (category: {details.category})" if details else "")
            )

        text = "".join(b.text for b in message.content if b.type == "text")
        if stream_callback and text:
            stream_callback(text)
        return LLMResponse(
            text=text.strip(),
            model=message.model,
            backend=self.name,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_s=latency,
            raw={"stop_reason": message.stop_reason},
        )


# --- Ollama (native API) ------------------------------------------------


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, *,
                 url: str = DEFAULT_OLLAMA_URL,
                 timeout: float = DEFAULT_TIMEOUT,
                 keep_alive: str = DEFAULT_KEEP_ALIVE,
                 stream: bool = True,
                 num_ctx: int = DEFAULT_NUM_CTX,
                 client: Optional[httpx.Client] = None):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        # Streaming is the default because a non-streamed request returns no
        # bytes at all until generation finishes: the read timeout then has to
        # cover model load + prompt eval + every output token, and a slow CPU
        # box fails with nothing recovered. Streaming resets the clock on each
        # token, so the timeout means "stalled", not "slower than expected".
        self.stream = stream
        # Context window requested from Ollama. Left unset, Ollama uses its
        # small default and silently drops the *start* of a longer prompt: the
        # grounding rules and the best evidence. It only ever grows (see
        # `_options`), because a different num_ctx per request makes Ollama
        # reload the whole model.
        self.num_ctx = num_ctx
        self._client = client or httpx.Client(timeout=_timeout(timeout))

    def _options(self, prompt_tokens: int, max_tokens: int, temperature: float) -> dict:
        # 25% headroom: the token estimate is approximate, and running out of
        # context truncates silently rather than failing.
        needed = int((prompt_tokens + max_tokens) * 1.25) + 64
        while needed > self.num_ctx and self.num_ctx < MAX_NUM_CTX:
            self.num_ctx *= 2
        return {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx}

    def warmup(self) -> float:
        """Load the model now rather than inside the first question.

        Returns seconds taken. An empty prompt makes Ollama load the weights
        and return immediately. It is loaded with the context size requests
        will use, or the first real request would reload it.
        """
        started = time.perf_counter()
        try:
            self._client.post(
                f"{self.url}/api/chat",
                json={"model": self.model, "messages": [], "stream": False,
                      "keep_alive": self.keep_alive,
                      "options": {"num_ctx": self.num_ctx}},
            )
        except httpx.HTTPError as exc:
            logger.warning("Warm-up failed for %s: %s", self.model, exc)
        return time.perf_counter() - started

    def available(self) -> bool:
        try:
            return self._client.get(f"{self.url}/api/tags", timeout=2.0).status_code == 200
        except Exception:
            return False

    def list_model_details(self) -> list[dict]:
        """Installed models as {"name", "size"}, smallest first.

        On a CPU, model size is the best predictor of answer latency, so the
        smallest model is the sensible default and is listed first.
        """
        response = self._client.get(f"{self.url}/api/tags", timeout=5.0)
        response.raise_for_status()
        models = [
            {"name": m["name"], "size": int(m.get("size") or 0)}
            for m in response.json().get("models", [])
        ]
        return sorted(models, key=lambda m: (m["size"] or float("inf"), m["name"]))

    def list_models(self) -> list[str]:
        return [m["name"] for m in self.list_model_details()]

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 stream_callback: Optional[Any] = None) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        prompt_tokens = estimate_tokens(prompt) + (estimate_tokens(system) if system else 0)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream or bool(stream_callback),
            "keep_alive": self.keep_alive,
            # temperature 0 matters here: grounded answers must be reproducible
            # across eval runs, and local models default to 0.8.
            "options": self._options(prompt_tokens, max_tokens, temperature),
        }

        started = time.perf_counter()
        try:
            if not self.stream and not stream_callback:
                response = self._client.post(f"{self.url}/api/chat", json=payload)
                response.raise_for_status()
                return self._to_response(response.json(), started)
            return self._complete_streaming(payload, started, stream_callback=stream_callback)
        except httpx.ReadTimeout as exc:
            raise LLMTimeout(
                f"{self.model} produced no output for {self.timeout:.0f}s. "
                f"On CPU an 8B model needs minutes for a first load — try "
                f"`ollama run {self.model}` once to warm it, shrink the prompt "
                f"via evidence_token_budget, or raise LLM_TIMEOUT."
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMTimeout(
                f"Could not reach Ollama at {self.url}. Is `ollama serve` running?"
            ) from exc

    def _complete_streaming(self, payload: dict, started: float,
                            stream_callback: Optional[Any] = None) -> LLMResponse:
        """Consume Ollama's newline-delimited JSON stream.

        Each token resets httpx's read clock, so a long generation no longer
        looks like a hang, and partial output survives a mid-stream failure.
        """
        pieces: list[str] = []
        final: dict[str, Any] = {}

        with self._client.stream("POST", f"{self.url}/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping unparseable stream line: %.80s", line)
                    continue
                delta = event.get("message", {}).get("content") or ""
                if delta:
                    pieces.append(delta)
                    if stream_callback:
                        stream_callback(delta)
                if event.get("done"):
                    final = event

        final["message"] = {"content": "".join(pieces)}
        return self._to_response(final, started)

    def _to_response(self, body: dict, started: float) -> LLMResponse:
        return LLMResponse(
            text=(body.get("message", {}).get("content") or "").strip(),
            model=body.get("model", self.model),
            backend=self.name,
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            latency_s=time.perf_counter() - started,
            raw={"done_reason": body.get("done_reason")},
        )


# --- OpenAI-compatible servers (Local / Groq) ---------------------------


class OpenAICompatibleBackend:
    """llama.cpp / LM Studio / vLLM / Ollama's /v1 endpoint / Groq.

    Names the wire format, not a vendor.
    """

    name = "openai"

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, *,
                 url: str = DEFAULT_LOCAL_URL,
                 api_key: str = "not-needed",
                 timeout: float = DEFAULT_TIMEOUT,
                 client: Optional[httpx.Client] = None):
        self.model = model
        self.url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=timeout)

    def available(self) -> bool:
        try:
            r = self._client.get(f"{self.url}/models", headers=self._headers, timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 stream_callback: Optional[Any] = None) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": bool(stream_callback),
        }

        started = time.perf_counter()
        if stream_callback:
            pieces: list[str] = []
            prompt_tokens = 0
            completion_tokens = 0
            finish_reason = None
            with self._client.stream("POST", f"{self.url}/chat/completions",
                                     json=payload, headers=self._headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}).get("content", "")
                        if delta:
                            pieces.append(delta)
                            stream_callback(delta)
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0].get("finish_reason")
                    usage = event.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

            full_text = "".join(pieces).strip()
            latency = time.perf_counter() - started
            return LLMResponse(
                text=full_text,
                model=self.model,
                backend=self.name,
                input_tokens=prompt_tokens or len(prompt.split()),
                output_tokens=completion_tokens or len(full_text.split()),
                latency_s=latency,
                raw={"finish_reason": finish_reason},
            )

        response = self._client.post(f"{self.url}/chat/completions",
                                     json=payload, headers=self._headers)
        response.raise_for_status()
        body = response.json()
        latency = time.perf_counter() - started

        choices = body.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = body.get("usage") or {}
        return LLMResponse(
            text=(text or "").strip(),
            model=body.get("model", self.model),
            backend=self.name,
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
            latency_s=latency,
            raw={"finish_reason": choices[0].get("finish_reason") if choices else None},
        )


class GroqBackend(OpenAICompatibleBackend):
    """Groq Cloud API (blazing fast LPU inference, free tier with llama-3.1-8b-instant)."""

    name = "groq"

    def __init__(self, model: str = DEFAULT_GROQ_MODEL, *,
                 api_key: Optional[str] = None,
                 timeout: float = 60.0,
                 client: Optional[httpx.Client] = None):
        key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        super().__init__(model=model, url=DEFAULT_GROQ_URL, api_key=key, timeout=timeout, client=client)

    def available(self) -> bool:
        auth = self._headers.get("Authorization", "").replace("Bearer ", "").strip()
        return bool(auth.startswith("gsk_"))


# --- selection ----------------------------------------------------------


class StubBackend:
    """Deterministic fake model — no weights, no server, no key.

    Exists so the full pipeline (retrieval -> rerank -> gate -> citation
    validation) can be exercised in CI and dry runs before any local model is
    installed. It quotes the first evidence chunk and cites [1], which is the
    shape a correct answer takes, so citation rendering and grounding checks
    all run for real.
    """

    name = "stub"

    def __init__(self, model: str = "stub"):
        self.model = model

    def available(self) -> bool:
        return True

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 stream_callback: Optional[Any] = None) -> LLMResponse:
        first = ""
        for line in prompt.splitlines():
            if line.startswith("[1] "):
                first = line[4:].strip()
                break
        text = f"{first} [1]" if first else "INSUFFICIENT_EVIDENCE"
        if stream_callback and text:
            stream_callback(text)
        return LLMResponse(
            text=text,
            model=self.model,
            backend=self.name,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            latency_s=0.0,
        )


class FallbackBackend:
    """A primary backend with a second one behind it.

    Groq answers in about a second but depends on the network and a free-tier
    rate limit; a local Ollama model is slow but always there. A transport
    failure or HTTP error from the primary is answered by the fallback, and
    `LLMResponse.backend` records which engine actually answered.
    """

    def __init__(self, primary: LLMBackend, fallback: LLMBackend):
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.model = primary.model

    def available(self) -> bool:
        return self.primary.available() or self.fallback.available()

    def warmup(self) -> float:
        # Only the primary: loading the fallback's weights would spend RAM and
        # seconds on an engine that is rarely needed.
        warm = getattr(self.primary, "warmup", None)
        return warm() if warm else 0.0

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 stream_callback: Optional[Any] = None) -> LLMResponse:
        streamed: list[str] = []

        def relay(token: str) -> None:
            streamed.append(token)
            stream_callback(token)

        kwargs = {"system": system, "max_tokens": max_tokens, "temperature": temperature}
        try:
            return self.primary.complete(
                prompt, stream_callback=relay if stream_callback else None, **kwargs
            )
        except (httpx.HTTPError, LLMTimeout) as exc:
            if streamed:
                # Part of the primary's answer is already on screen; splicing a
                # different model's answer after it is worse than failing.
                raise
            logger.warning("%s failed (%s); answering with %s instead.",
                           self.primary.name, exc, self.fallback.name)
            return self.fallback.complete(prompt, stream_callback=stream_callback, **kwargs)


BACKENDS = {
    "anthropic": AnthropicBackend,
    "groq": GroqBackend,
    "ollama": OllamaBackend,
    "openai": OpenAICompatibleBackend,
    "stub": StubBackend,
}


def _real_anthropic_key() -> Optional[str]:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    return key if key.startswith("sk-ant-") else None


def _real_groq_key() -> Optional[str]:
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    return key if key.startswith("gsk_") else None


def available_backends() -> dict[str, bool]:
    """Probe every backend. Useful for diagnostics and for `check_llm.py`."""
    status = {}
    for name, cls in BACKENDS.items():
        try:
            status[name] = cls().available()
        except Exception:
            status[name] = False
    return status


def get_llm(backend: Optional[str] = None, *, model: Optional[str] = None) -> LLMBackend:
    """Return a backend by name, or auto-select one.

    Resolution order for "auto": a real Groq key or Anthropic key wins,
    otherwise a running local server is used.
    """
    backend = (backend or os.environ.get("RAG_LLM_BACKEND") or "auto").lower()

    if backend != "auto":
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. Choose from {sorted(BACKENDS)} or 'auto'."
            )
        cls = BACKENDS[backend]
        return cls(model) if model else cls()

    if _real_groq_key():
        logger.info("Auto-selected the Groq backend (free high-speed cloud inference).")
        return GroqBackend(model or DEFAULT_GROQ_MODEL)

    if _real_anthropic_key():
        logger.info("Auto-selected the anthropic backend.")
        return AnthropicBackend(model or DEFAULT_ANTHROPIC_MODEL)

    # "stub" is deliberately excluded from auto-selection: a fake answer must
    # never appear in results by default. Ask for it explicitly.
    for name in ("ollama", "openai"):
        candidate = BACKENDS[name](model) if model else BACKENDS[name]()
        if candidate.available():
            logger.info("Auto-selected the %s backend.", name)
            return candidate

    raise RuntimeError(
        "No LLM backend available. Either set a GROQ_API_KEY (gsk_...), an ANTHROPIC_API_KEY (sk-ant-...), "
        "or start a local model:\n"
        "  ollama serve && ollama pull qwen2.5:1.5b       # then RAG_LLM_BACKEND=ollama\n"
        "  llama-server -m model.gguf --port 8080         # then RAG_LLM_BACKEND=openai\n"
        "Run `python eval/check_llm.py --list` to see what is reachable."
    )
