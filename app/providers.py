"""Provider-agnostic factories for the chat model, embeddings, and re-ranker.

Everything downstream talks to LangChain interfaces, so swapping OpenAI for a
local model is a single env var (LLM_PROVIDER) and never a code change.
"""

from __future__ import annotations

import functools

import numpy as np
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.embeddings import Embeddings

import config


class FastEmbedEmbeddings(Embeddings):
    """Local embeddings on ONNX Runtime via fastembed - no PyTorch.

    BGE v1.5 expects an instruction prefix on the query side only, so queries
    and passages take different paths here rather than relying on the
    library's per-model defaults.
    """

    def __init__(self, model: str, query_prompt: str = "", normalize: bool = True):
        from fastembed import TextEmbedding

        self.model = model
        self._model = TextEmbedding(model_name=model, cache_dir=config.MODEL_CACHE_DIR)
        self._query_prompt = query_prompt
        self._normalize = normalize

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = np.asarray(list(self._model.embed(texts, batch_size=64)), dtype=np.float32)
        if self._normalize:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prompt + text])[0]


class FastEmbedCrossEncoder(BaseCrossEncoder):
    """Cross-encoder re-ranker on ONNX Runtime, pluggable into LangChain's
    CrossEncoderReranker."""

    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = model
        self._model = TextCrossEncoder(model_name=model, cache_dir=config.MODEL_CACHE_DIR)

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self._model.rerank_pairs(list(text_pairs))]


def embedding_name(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Resolve (provider, model), filling gaps from config."""
    provider = (provider or config.EMBED_PROVIDER).strip().lower()
    if model is None:
        model = config.EMBEDDING_MODEL if provider == config.EMBED_PROVIDER else config.default_embedding_model(provider)
    return provider, model


@functools.lru_cache(maxsize=4)
def get_embeddings(provider: str | None = None, model: str | None = None):
    """Embeddings follow EMBED_PROVIDER, which defaults to LLM_PROVIDER.

    Parameterised so the eval harness can compare embedding models in one run.
    """
    provider, model = embedding_name(provider, model)
    config.require_embed_key(provider)

    if provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        size = int(model.rsplit("-", 1)[-1]) if model.rsplit("-", 1)[-1].isdigit() else 1536
        return DeterministicFakeEmbedding(size=size)

    if provider == "local":
        # BGE models are trained for cosine similarity, so unit-norm vectors
        # are what FAISS's L2 distance should be comparing.
        prompt = config.BGE_QUERY_PROMPT if "bge" in model.lower() else ""
        return FastEmbedEmbeddings(model, query_prompt=prompt, normalize=config.EMBED_NORMALIZE)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=config.OPENAI_API_KEY)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=config.GOOGLE_API_KEY)


@functools.lru_cache(maxsize=2)
def get_cross_encoder(model: str | None = None):
    """Loaded once per process: a cross-encoder is hundreds of MB of weights."""
    return FastEmbedCrossEncoder(model or config.RERANK_MODEL)


@functools.lru_cache(maxsize=1)
def get_chat_model(streaming: bool = False):
    if config.PROVIDER == "local":
        from pathlib import Path

        from langchain_community.chat_models import ChatLlamaCpp

        model_path = Path(config.CHAT_MODEL)
        if not model_path.is_file():
            raise SystemExit(
                f"Local chat model not found at {model_path.resolve()}. Install the extra "
                "dependency (pip install -r requirements-local-llm.txt), then download it with:"
                "    python scripts/fetch_local_model.py"
            )
        return ChatLlamaCpp(
            model_path=str(model_path),
            n_ctx=config.LLAMA_N_CTX,
            n_threads=config.LLAMA_N_THREADS,
            n_batch=config.LLAMA_N_BATCH,
            max_tokens=config.LLAMA_MAX_TOKENS,
            temperature=config.TEMPERATURE,
            verbose=False,
        )

    config.require_api_key()
    if config.PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.CHAT_MODEL,
            temperature=config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY,
            streaming=streaming,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        temperature=config.TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
        # Gemini streams fine without an explicit flag; kept for parity.
    )
