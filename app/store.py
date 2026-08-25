"""Build, persist, and load the FAISS index (plus the chunk sidecar for BM25)."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from collections import deque
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
from app.providers import get_embeddings

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"


_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", re.IGNORECASE)


class _RateLimiter:
    """Keep any rolling 60-second window under a requests-per-minute ceiling.

    The provider bills one request per *document*, not per batch, so a batch of
    64 spends 64 units at once. Two things make naive pacing fail here:

    1. A batch is a burst, not a stream. Spacing batches by their average cost
       still puts two bursts inside one 60s window, which overshoots.
    2. A retry re-sends the whole batch, spending the quota again. Without a
       window that actually drains, retries feed the very limit they are
       waiting on, and the run dies after burning every attempt.

    So we track individual request timestamps and block until enough of them
    have aged out of the window.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._times: deque[float] = deque()

    def acquire(self, units: int) -> None:
        if self.rpm <= 0:
            return
        while True:
            now = time.monotonic()
            cutoff = now - 60.0
            while self._times and self._times[0] < cutoff:
                self._times.popleft()
            if len(self._times) + units <= self.rpm or not self._times:
                break
            # Sleep until the oldest request leaves the window.
            time.sleep(max(0.1, self._times[0] - cutoff + 0.05))
        now = time.monotonic()
        self._times.extend([now] * units)


def _retry_delay_from(message: str) -> float | None:
    match = _RETRY_DELAY_RE.search(message)
    if not match:
        return None
    return float(match.group(1) or match.group(2))


def _embed_with_retry(embeddings, batch: list[str], limiter: "_RateLimiter | None" = None) -> list[list[float]]:
    """Free-tier embedding endpoints rate-limit aggressively; pace, then retry."""
    delay = 2.0
    for attempt in range(config.EMBED_MAX_RETRIES):
        if limiter:
            limiter.acquire(len(batch))
        try:
            return embeddings.embed_documents(batch)
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            retriable = any(
                t in lowered for t in ("rate", "429", "quota", "resource_exhausted", "timeout", "503", "unavailable")
            )
            if not retriable or attempt == config.EMBED_MAX_RETRIES - 1:
                if "quota" in lowered or "429" in lowered:
                    raise SystemExit(
                        "Embedding quota exhausted and retries gave up.\n"
                        "  - lower the rate with EMBED_RPM (currently "
                        f"{config.EMBED_RPM}) in .env, or\n"
                        "  - wait for the daily quota to reset, or\n"
                        "  - switch provider with LLM_PROVIDER=openai.\n"
                        f"Provider said: {message[:300]}"
                    ) from exc
                raise
            # Prefer the server's own retry hint over our guess.
            sleep_for = (_retry_delay_from(message) or delay) + random.uniform(0, 1)
            logger.warning(
                "Embedding batch failed (%s); retrying in %.1fs [attempt %d/%d]",
                exc.__class__.__name__,
                sleep_for,
                attempt + 1,
                config.EMBED_MAX_RETRIES,
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")  # pragma: no cover


def _cache_path() -> Path:
    slug = config.EMBEDDING_MODEL.replace("/", "_")
    return Path(config.EMBED_CACHE_DIR) / f"{slug}.jsonl"


def _load_cache() -> dict[str, list[float]]:
    """Vectors already paid for, keyed by a hash of the exact text."""
    path = _cache_path()
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
                cache[record["k"]] = record["v"]
            except (json.JSONDecodeError, KeyError):
                continue  # a torn final line from an interrupted run
    return cache


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_index(chunks: list[Document], show_progress: bool = True) -> FAISS:
    """Embed every chunk in batches and assemble a FAISS index.

    Vectors are cached to disk as they arrive. A run killed by a rate limit or
    a lost connection therefore resumes from where it stopped instead of
    re-spending quota on work already done.
    """
    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    cache = _load_cache() if config.EMBED_CACHE else {}
    todo = [t for t in texts if _text_key(t) not in cache]
    if cache:
        logger.info("Embedding cache: %d/%d chunks already embedded", len(texts) - len(todo), len(texts))

    if todo:
        cache_file = None
        if config.EMBED_CACHE:
            _cache_path().parent.mkdir(parents=True, exist_ok=True)
            cache_file = _cache_path().open("a", encoding="utf-8")

        batch_size = max(1, config.EMBED_BATCH_SIZE)
        batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]

        iterator = batches
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(batches, desc="Embedding", unit="batch")
            except ImportError:
                pass

        limiter = _RateLimiter(config.EMBED_RPM)
        try:
            for batch in iterator:
                for text, vector in zip(batch, _embed_with_retry(embeddings, batch, limiter)):
                    cache[_text_key(text)] = vector
                    if cache_file:
                        cache_file.write(json.dumps({"k": _text_key(text), "v": vector}) + "\n")
                if cache_file:
                    cache_file.flush()  # survive a kill between batches
        finally:
            if cache_file:
                cache_file.close()

    vectors = [cache[_text_key(t)] for t in texts]

    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedded {len(vectors)} vectors for {len(texts)} chunks")

    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=metadatas,
    )


def save_index(store: FAISS, chunks: list[Document], index_dir: str | None = None) -> Path:
    path = Path(index_dir or config.INDEX_DIR)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))

    with (path / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps({"text": chunk.page_content, "metadata": chunk.metadata}) + "\n")

    (path / META_FILE).write_text(
        json.dumps(
            {
                "embed_provider": config.EMBED_PROVIDER,
                "chat_provider": config.PROVIDER,
                "embedding_model": config.EMBEDDING_MODEL,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "chunk_count": len(chunks),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_chunks(index_dir: str | None = None) -> list[Document]:
    path = Path(index_dir or config.INDEX_DIR) / CHUNKS_FILE
    if not path.exists():
        return []
    docs: list[Document] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            docs.append(Document(page_content=record["text"], metadata=record["metadata"]))
    return docs


def load_index(index_dir: str | None = None) -> FAISS:
    path = Path(index_dir or config.INDEX_DIR)
    if not (path / "index.faiss").exists():
        raise SystemExit(f"No index at {path.resolve()}. Build one first:\n\n    python -m app.ingest\n")

    meta_path = path / META_FILE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedding_model") != config.EMBEDDING_MODEL:
            raise SystemExit(
                f"Index was built with embedding model '{meta.get('embedding_model')}' but the current "
                f"config says '{config.EMBEDDING_MODEL}'. Vectors from different models are not "
                f"comparable - rebuild with: python -m app.ingest --rebuild"
            )

    # Safe: we only ever load an index this project wrote itself.
    return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)


def index_meta(index_dir: str | None = None) -> dict:
    path = Path(index_dir or config.INDEX_DIR) / META_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
