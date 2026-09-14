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
# "openai" (default), "local", or "google".
#
# EMBED_PROVIDER is separate on purpose: embedding and chat have very different
# cost shapes. Embedding is a one-time bulk job over the whole corpus and is the
# thing that hits quota walls; chat is one call per question and rarely does.
# Running embeddings locally while chat stays on a hosted model is a sensible
# mix when re-indexing often.
PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embeddings default to whichever provider handles chat, unless overridden.
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", PROVIDER).strip().lower()

_DEFAULT_MODELS = {
    "openai": ("gpt-4o-mini", "text-embedding-3-small"),
    # Embedding chosen on the dev set: with SPLADE it gave the best ranking
    # measured (MRR 0.829, hit@1 78%, eval-dev-20260914-154243.json).
    "local": ("models/qwen2.5-3b-instruct-q4_k_m.gguf", "snowflake/snowflake-arctic-embed-m"),
    "google": ("gemini-flash-latest", "models/gemini-embedding-001"),
    # Deterministic hash-seeded vectors: no model, no network. Used by the scale
    # benchmark to exercise everything except embedding quality.
    "fake": (None, "fake-1536"),
}


def default_embedding_model(provider: str) -> str:
    return _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["openai"])[1]


CHAT_MODEL = os.getenv("CHAT_MODEL", _DEFAULT_MODELS.get(PROVIDER, _DEFAULT_MODELS["openai"])[0] or "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", default_embedding_model(EMBED_PROVIDER))

# Local embedding and re-ranking run on ONNX Runtime (fastembed), not PyTorch:
# a fraction of the install size, and it loads on machines whose application
# control policy blocks torch's unsigned DLLs.
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", ".cache/models")
EMBED_NORMALIZE = _bool("EMBED_NORMALIZE", True)
# BGE v1.5 is trained with this instruction on the *query* side only; passages
# are embedded bare. Leaving it off costs retrieval quality on short questions.
# Applied only to models whose name contains "bge"; set to "" to disable.
BGE_QUERY_PROMPT = os.getenv("BGE_QUERY_PROMPT", "Represent this sentence for searching relevant passages: ")

# Local chat knobs (llama.cpp). n_ctx must hold the system prompt plus TOP_K
# passages plus the answer; 4096 is comfortable for 5 chunks of ~1000 chars.
LLAMA_N_CTX = _int("LLAMA_N_CTX", 4096)
LLAMA_N_THREADS = _int("LLAMA_N_THREADS", os.cpu_count() or 4)
LLAMA_N_BATCH = _int("LLAMA_N_BATCH", 512)
LLAMA_MAX_TOKENS = _int("LLAMA_MAX_TOKENS", 512)

# --- citations ----------------------------------------------------------------
# "model"  trust the model to emit [n] markers (fine for hosted models)
# "auto"   fall back to computed attribution when the model emits none
# "off"    never annotate; the source panel still lists what was retrieved
# Local models below ~7B routinely ignore the citation instruction, so "auto"
# is the default there.
CITATION_MODE = os.getenv("CITATION_MODE", "auto" if PROVIDER == "local" else "model").strip().lower()
CITE_THRESHOLD = _float("CITE_THRESHOLD", 0.55)
CITE_MIN_CHARS = _int("CITE_MIN_CHARS", 40)
TEMPERATURE = _float("TEMPERATURE", 0.0)

# --- paths --------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")

# --- ingestion: loading -------------------------------------------------------
# Worker processes for document loading. PDF text extraction is the slow stage
# on a 10k-page corpus and parallelises cleanly per file. 0 = auto (one worker
# per core, but single-process for small corpora where spawn cost dominates).
INGEST_WORKERS = _int("INGEST_WORKERS", 0)

# --- chunking -----------------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
MIN_CHUNK_CHARS = _int("MIN_CHUNK_CHARS", 80)  # drop near-empty fragments
# Contextual header prepended to every chunk before embedding:
#   "path"        title + section breadcrumb, e.g. [tutorial request files > Request Files > UploadFile]
#   "path-clean"  breadcrumb minus headings that recur across documents ("Recap", "Check it")
#   "title"       document title only, e.g. [tutorial request files]
#   "none"        no header
# Default measured on the dev set (eval-dev-20260914-084331.json): path-clean
# kept hit@5 at 32/36 and raised MRR 0.762 -> 0.796 over path.
HEADER_MODE = os.getenv("HEADER_MODE", "path-clean").strip().lower()
# path-clean: a heading found in at least this many documents is boilerplate.
BOILERPLATE_HEADING_MIN_DOCS = _int("BOILERPLATE_HEADING_MIN_DOCS", 3)

# --- retrieval ----------------------------------------------------------------
TOP_K = _int("TOP_K", 5)  # chunks handed to the LLM
FETCH_K = _int("FETCH_K", 20)  # candidates pulled before MMR re-ranking
# 1.0 = pure relevance, 0.0 = pure diversity.
# Measured on the FastAPI corpus (eval --ablate, 36 questions): dropping to 0.5
# cost 5.5 points of hit@5 (83.3% -> 77.8%) and 20 points of precision@5
# (47.2% -> 27.2%). Diversity re-ranking was removing relevant chunks, not
# redundant ones, because sibling chunks of one long page are often all needed.
# Lower it only if your corpus has genuine near-duplicate documents.
MMR_LAMBDA = _float("MMR_LAMBDA", 1.0)
USE_HYBRID = _bool("USE_HYBRID", True)  # BM25 + dense ensemble
HYBRID_WEIGHTS = (
    _float("WEIGHT_DENSE", 0.6),
    _float("WEIGHT_SPARSE", 0.4),
)
# Cross-encoder re-ranking: pull RERANK_FETCH_K candidates from the first-stage
# retriever, score each (question, passage) pair jointly, keep the best TOP_K.
RERANK = _bool("RERANK", False)
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_FETCH_K = _int("RERANK_FETCH_K", 30)
# Order by reciprocal rank fusion of first-stage and re-ranker ranks, rather
# than the re-ranker's order alone.
RERANK_FUSION = _bool("RERANK_FUSION", True)
# Lexical leg of hybrid retrieval: "splade" (learned sparse: weights terms a
# passage implies but does not contain) or "bm25". SPLADE ranked better on the
# dev set but costs a model pass per chunk at ingest; for very large corpora on
# CPU, SPARSE=bm25 ingests orders of magnitude faster.
SPARSE = os.getenv("SPARSE", "splade").strip().lower()
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "prithivida/Splade_PP_en_v1")

# --- ingestion: embedding -----------------------------------------------------
# Local embedding has no quota and no per-request overhead, so it wants big
# batches; hosted providers want small ones so a burst cannot overshoot a cap.
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 64 if EMBED_PROVIDER == "local" else 100)
EMBED_MAX_RETRIES = _int("EMBED_MAX_RETRIES", 8)
# Requests per minute to allow ourselves. The provider counts one request per
# document, not per batch, so this throttles documents. Google's free tier caps
# embedding at 100/min; 90 leaves headroom for clock skew.
# 0 disables throttling entirely - correct for local models and for OpenAI,
# whose paid-tier limits are far above anything a single ingest reaches.
EMBED_RPM = _int("EMBED_RPM", 90 if EMBED_PROVIDER == "google" else 0)
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


def require_embed_key(provider: str | None = None) -> None:
    """Embedding-only entry points (ingest, eval) need no key when local."""
    provider = provider or EMBED_PROVIDER
    if provider in {"local", "fake"}:
        return
    if provider == "openai" and not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set (embedding provider is openai).")
    if provider == "google" and not GOOGLE_API_KEY:
        raise SystemExit("GOOGLE_API_KEY is not set (embedding provider is google).")


def summary() -> str:
    return (
        f"chat={PROVIDER}:{CHAT_MODEL} embed={EMBED_PROVIDER}:{EMBEDDING_MODEL} "
        f"chunk={CHUNK_SIZE}/{CHUNK_OVERLAP} header={HEADER_MODE} top_k={TOP_K} "
        f"hybrid={USE_HYBRID} rerank={RERANK_MODEL if RERANK else 'off'}"
    )
