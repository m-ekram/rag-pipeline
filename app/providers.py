"""Provider-agnostic factories for the chat model and the embedding model.

Everything downstream talks to LangChain interfaces, so swapping Gemini for
OpenAI is a single env var (LLM_PROVIDER) and never a code change.
"""

from __future__ import annotations

import functools

import config


@functools.lru_cache(maxsize=1)
def get_embeddings():
    """Embeddings follow EMBED_PROVIDER, which defaults to LLM_PROVIDER."""
    config.require_embed_key()

    if config.EMBED_PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBED_DEVICE},
            # BGE/E5 models are trained for cosine similarity, so unit-norm
            # vectors are what FAISS's inner product should be comparing.
            encode_kwargs={"normalize_embeddings": config.EMBED_NORMALIZE},
        )

    if config.EMBED_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )


@functools.lru_cache(maxsize=1)
def get_chat_model(streaming: bool = False):
    if config.PROVIDER == "local":
        from pathlib import Path

        from langchain_community.chat_models import ChatLlamaCpp

        model_path = Path(config.CHAT_MODEL)
        if not model_path.is_file():
            raise SystemExit(
                f"Local chat model not found at {model_path.resolve()}. "
                "Download it with:    python scripts/fetch_local_model.py"
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
