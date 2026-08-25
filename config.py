"""Central configuration, loaded from the environment (.env)."""

import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- provider -----------------------------------------------------------------
# "google" (default, has a free tier), "openai", or "local".
#
# EMBED_PROVIDER is separate on purpose: embedding and chat have very different
# cost shapes. Embedding is a one-time bulk job over the whole corpus and is the
# thing that hits quota walls; chat is one call per question and rarely does.
# Running embeddings locally while chat stays on a hosted model is the
# combination that makes this project free to re-index as often as you like.
PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embeddings default to whichever provider handles chat, unless overridden.
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", PROVIDER).strip().lower()

_DEFAULT_MODELS = {
    "google": ("gemini-flash-latest", "models/gemini-embedding-001"),
    "openai": ("gpt-4o-mini", "text-embedding-3-small"),
    "local": ("models/qwen2.5-3b-instruct-q4_k_m.gguf", "BAAI/bge-small-en-v1.5"),
}
_chat_default = _DEFAULT_MODELS.get(PROVIDER, _DEFAULT_MODELS["google"])[0]
_embed_default = _DEFAULT_MODELS.get(EMBED_PROVIDER, _DEFAULT_MODELS["google"])[1]

CHAT_MODEL = os.getenv("CHAT_MODEL", _chat_default or "gemini-flash-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _embed_default)

# Local embedding knobs (ignored by hosted providers).
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED_NORMALIZE = _bool("EMBED_NORMALIZE", True)

# Local chat knobs (llama.cpp). n_ctx must hold the system prompt plus TOP_K
# passages plus the answer; 4096 is comfortable for 5 chunks of ~1000 chars.
LLAMA_N_CTX = _int("LLAMA_N_CTX", 4096)
LLAMA_N_THREADS = _int("LLAMA_N_THREADS", os.cpu_count() or 4)
LLAMA_N_BATCH = _int("LLAMA_N_BATCH", 512)
LLAMA_MAX_TOKENS = _int("LLAMA_MAX_TOKENS", 512)
TEMPERATURE = _float("TEMPERATURE", 0.0)

# --- paths --------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")

# --- chunking -----------------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
MIN_CHUNK_CHARS = _int("MIN_CHUNK_CHARS", 80)  # drop near-empty fragments

# --- retrieval ----------------------------------------------------------------
TOP_K = _int("TOP_K", 5)  # chunks handed to the LLM
FETCH_K = _int("FETCH_K", 20)  # candidates pulled before MMR re-ranking
MMR_LAMBDA = _float("MMR_LAMBDA", 0.5)  # 1.0 = pure relevance, 0.0 = pure diversity
USE_HYBRID = _bool("USE_HYBRID", True)  # BM25 + dense ensemble
HYBRID_WEIGHTS = (
    _float("WEIGHT_DENSE", 0.6),
    _float("WEIGHT_SPARSE", 0.4),
)

# --- ingestion ----------------------------------------------------------------
# Local embedding has no quota and no per-request overhead, so it wants big
# batches; hosted providers want small ones so a burst cannot overshoot a cap.
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 64 if EMBED_PROVIDER == "local" else 20)
EMBED_MAX_RETRIES = _int("EMBED_MAX_RETRIES", 8)
# Requests per minute to allow ourselves. The provider counts one request per
# document, not per batch, so this throttles documents. Google's free tier caps
# embedding at 100/min; 90 leaves headroom for clock skew.
# 0 disables throttling entirely - correct for local models, which answer from
# your own CPU and have no rate limit to respect.
EMBED_RPM = _int("EMBED_RPM", 0 if EMBED_PROVIDER == "local" else 90)
# Cache vectors on disk so an interrupted ingest resumes instead of re-paying.
EMBED_CACHE = _bool("EMBED_CACHE", True)
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", ".embed_cache")

# --- serving ------------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = _int("PORT", 8000)
MAX_HISTORY_TURNS = _int("MAX_HISTORY_TURNS", 6)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


def active_api_key() -> str | None:
    return OPENAI_API_KEY if PROVIDER == "openai" else GOOGLE_API_KEY


def require_api_key() -> None:
    """Fail fast with an actionable message instead of a deep SDK traceback."""
    if PROVIDER == "local":
        return  # runs on your own CPU; there is no key to check
    if active_api_key():
        return
    if PROVIDER == "openai":
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://platform.openai.com/api-keys"
        )
    raise SystemExit(
        "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key "
        "from https://aistudio.google.com/apikey"
    )


def require_embed_key() -> None:
    """Embedding-only entry points (ingest, eval) need no key when local."""
    if EMBED_PROVIDER == "local":
        return
    if EMBED_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set (EMBED_PROVIDER=openai).")
    if EMBED_PROVIDER == "google" and not GOOGLE_API_KEY:
        raise SystemExit("GOOGLE_API_KEY is not set (EMBED_PROVIDER=google).")


def summary() -> str:
    return (
        f"chat={PROVIDER}:{CHAT_MODEL} embed={EMBED_PROVIDER}:{EMBEDDING_MODEL} "
        f"chunk={CHUNK_SIZE}/{CHUNK_OVERLAP} top_k={TOP_K} hybrid={USE_HYBRID}"
    )
