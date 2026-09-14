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


def prompts_for(model: str) -> tuple[str, str]:
    """(query prefix, document prefix) each model was trained with, per its model card.

    Retrieval models are trained asymmetrically; embedding a query without its
    instruction (or a passage without its marker) costs recall.
    """
    name = model.lower()
    table = [
        ("bge", (config.BGE_QUERY_PROMPT, "")),
        ("snowflake-arctic-embed", ("Represent this sentence for searching relevant passages: ", "")),
        ("mxbai-embed", ("Represent this sentence for searching relevant passages: ", "")),
        ("nomic-embed-text", ("search_query: ", "search_document: ")),
        ("e5", ("query: ", "passage: ")),
    ]
    return next((prompts for key, prompts in table if key in name), ("", ""))


class FastEmbedEmbeddings(Embeddings):
    """Local embeddings on ONNX Runtime via fastembed - no PyTorch.

    Queries and passages take different paths so each model gets the prefixes
    it was trained with (see `prompts_for`), rather than relying on the
    library's per-model defaults.
    """

    def __init__(self, model: str, query_prompt: str = "", normalize: bool = True, doc_prompt: str = ""):
        from fastembed import TextEmbedding

        self.model = model
        self._model = TextEmbedding(model_name=model, cache_dir=config.MODEL_CACHE_DIR)
        self._query_prompt = query_prompt
        self._doc_prompt = doc_prompt
        self._normalize = normalize

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = np.asarray(list(self._model.embed(texts, batch_size=64)), dtype=np.float32)
        if self._normalize:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._doc_prompt + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prompt + text])[0]


class FastEmbedCrossEncoder(BaseCrossEncoder):
    """Cross-encoder re-ranker on ONNX Runtime: reads question and passage
    together and emits one relevance score per pair."""

    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = model
        self._model = TextCrossEncoder(model_name=model, cache_dir=config.MODEL_CACHE_DIR)

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self._model.rerank_pairs(list(text_pairs))]


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """ColBERT relevance: each query token takes its best-matching passage
    token; the scores are summed."""
    return float((np.asarray(query_tokens) @ np.asarray(doc_tokens).T).max(axis=1).sum())


class LateInteractionReranker(BaseCrossEncoder):
    """ColBERT-style re-ranker on ONNX Runtime.

    Question and passage are embedded separately, one vector per token, and
    compared token by token (MaxSim). Finer-grained than a single-vector
    embedding, and unlike a cross-encoder it scores exact token matches
    directly rather than judging the passage as a whole.
    """

    def __init__(self, model: str):
        from fastembed import LateInteractionTextEmbedding

        self.model = model
        self._model = LateInteractionTextEmbedding(model_name=model, cache_dir=config.MODEL_CACHE_DIR)

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        pairs = list(text_pairs)
        queries: dict[str, np.ndarray] = {}
        for query, _ in pairs:
            if query not in queries:
                queries[query] = next(iter(self._model.query_embed(query)))
        passages = self._model.embed([passage for _, passage in pairs], batch_size=32)
        return [maxsim(queries[query], tokens) for (query, _), tokens in zip(pairs, passages)]


def _is_late_interaction(model: str) -> bool:
    from fastembed import LateInteractionTextEmbedding

    return any(m["model"] == model for m in LateInteractionTextEmbedding.list_supported_models())


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
        query_prompt, doc_prompt = prompts_for(model)
        return FastEmbedEmbeddings(model, query_prompt=query_prompt, doc_prompt=doc_prompt, normalize=config.EMBED_NORMALIZE)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=config.OPENAI_API_KEY)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=config.GOOGLE_API_KEY)


@functools.lru_cache(maxsize=2)
def get_cross_encoder(model: str | None = None):
    """Loaded once per process: a re-ranker is hundreds of MB of weights.
    ColBERT-family models get MaxSim scoring; the rest are cross-encoders."""
    model = model or config.RERANK_MODEL
    return LateInteractionReranker(model) if _is_late_interaction(model) else FastEmbedCrossEncoder(model)


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
