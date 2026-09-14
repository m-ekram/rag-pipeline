"""Build, persist, and load the FAISS index (plus the chunk sidecar for BM25)."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
import uuid
from collections import deque
from pathlib import Path

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
from app.providers import get_embeddings

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"

# Vectors are moved into FAISS in blocks, so peak memory is the index itself
# plus one block - never a second full copy of the corpus.
_ADD_BLOCK = 4096

_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", re.IGNORECASE)
_KEY_RE = re.compile(r"^[0-9a-f]{40}$")


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
    """Hosted embedding endpoints rate-limit aggressively; pace, then retry."""
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
                        "  - wait for the quota to reset, or\n"
                        "  - switch provider with EMBED_PROVIDER=local.\n"
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


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _slug(model_name: str) -> str:
    return re.sub(r"[^\w.-]", "_", model_name)


class EmbedCache:
    """Vectors already paid for, keyed by a hash of the exact chunk text.

    On disk: `keys.txt` (one hash per line) and `vectors.f32` (raw float32
    rows, same order). The old format was JSONL of Python float lists; loading
    that for a 10k-page corpus (~40k chunks x 1536 dims) materialised ~60M
    Python floats, about 2 GB of heap before FAISS saw a single vector. Here
    only the key->row dict lives in RAM and rows are read through a memory map.

    Appends are flushed per batch, so a run killed by a rate limit resumes from
    the last completed batch. A torn tail from a killed run is trimmed on open.
    With `enabled=False` the same interface is kept purely in memory.
    """

    def __init__(self, model_name: str, root: str | Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self.dir = Path(root or config.EMBED_CACHE_DIR) / _slug(model_name)
        self._legacy = Path(root or config.EMBED_CACHE_DIR) / f"{model_name.replace('/', '_')}.jsonl"
        self.index: dict[str, int] = {}
        self.rows = 0
        self.dim: int | None = None
        self._memory: list[np.ndarray] = []
        if enabled:
            self._open()

    @property
    def _keys_path(self) -> Path:
        return self.dir / "keys.txt"

    @property
    def _vectors_path(self) -> Path:
        return self.dir / "vectors.f32"

    @property
    def _meta_path(self) -> Path:
        return self.dir / "meta.json"

    def _open(self) -> None:
        if not self._meta_path.exists():
            if self._legacy.exists():
                self._migrate_legacy()
            return
        self.dim = json.loads(self._meta_path.read_text(encoding="utf-8"))["dim"]
        keys = []
        if self._keys_path.exists():
            keys = [k for k in self._keys_path.read_text(encoding="utf-8").split("\n") if _KEY_RE.match(k)]
        row_bytes = 4 * self.dim
        on_disk = self._vectors_path.stat().st_size // row_bytes if self._vectors_path.exists() else 0
        n = min(len(keys), on_disk)
        if n != len(keys) or self._vectors_path.exists() and self._vectors_path.stat().st_size != n * row_bytes:
            logger.warning("Embedding cache: trimming torn tail (%d keys, %d vectors) to %d rows", len(keys), on_disk, n)
            with self._vectors_path.open("r+b") as fh:
                fh.truncate(n * row_bytes)
            self._keys_path.write_text("".join(f"{k}\n" for k in keys[:n]), encoding="utf-8")
        self.index = {k: i for i, k in enumerate(keys[:n])}
        self.rows = n

    def _migrate_legacy(self) -> None:
        """One-time streaming conversion from the old JSONL cache."""
        logger.info("Embedding cache: migrating %s to float32 storage", self._legacy.name)
        keys: list[str] = []
        vectors: list[list[float]] = []
        with self._legacy.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                    keys.append(record["k"])
                    vectors.append(record["v"])
                except (json.JSONDecodeError, KeyError):
                    continue  # a torn final line from an interrupted run
                if len(keys) >= 1000:
                    self.add(keys, np.asarray(vectors, dtype=np.float32))
                    keys, vectors = [], []
        if keys:
            self.add(keys, np.asarray(vectors, dtype=np.float32))

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def add(self, keys: list[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(keys):
            raise ValueError(f"Expected {len(keys)} vectors, got array of shape {vectors.shape}")
        if self.dim is None:
            self.dim = int(vectors.shape[1])
            if self.enabled:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._meta_path.write_text(json.dumps({"dim": self.dim}), encoding="utf-8")
        elif vectors.shape[1] != self.dim:
            raise ValueError(f"Cache holds {self.dim}-d vectors; got {vectors.shape[1]}-d")

        if self.enabled:
            # Vectors before keys: a kill between the two leaves an orphan row,
            # which _open trims, never a key pointing at a missing row.
            with self._vectors_path.open("ab") as fh:
                fh.write(vectors.tobytes())
            with self._keys_path.open("a", encoding="utf-8") as fh:
                fh.write("".join(f"{k}\n" for k in keys))
        else:
            self._memory.append(vectors)

        for i, key in enumerate(keys):
            self.index[key] = self.rows + i
        self.rows += len(keys)

    def get(self, keys: list[str]) -> np.ndarray:
        rows = np.fromiter((self.index[k] for k in keys), dtype=np.int64, count=len(keys))
        if self.enabled:
            matrix = np.memmap(self._vectors_path, dtype=np.float32, mode="r", shape=(self.rows, self.dim))
        else:
            matrix = np.concatenate(self._memory) if len(self._memory) > 1 else self._memory[0]
            self._memory = [matrix]
        return np.asarray(matrix[rows], dtype=np.float32)


def build_index(
    chunks: list[Document],
    show_progress: bool = True,
    embeddings=None,
    model_name: str | None = None,
) -> FAISS:
    """Embed every chunk in batches and assemble a FAISS index.

    `embeddings`/`model_name` default to the configured provider; the eval
    harness passes others to compare models. The cache is keyed by model, since
    vectors from different models are not comparable.
    """
    if not chunks:
        raise SystemExit("Nothing to index: chunking produced 0 chunks.")
    if embeddings is None:
        embeddings = get_embeddings()
        model_name = model_name or config.EMBEDDING_MODEL
    model_name = model_name or getattr(embeddings, "model", None) or getattr(embeddings, "model_name", "unknown")

    texts = [c.page_content for c in chunks]
    keys = [_text_key(t) for t in texts]
    cache = EmbedCache(model_name, enabled=config.EMBED_CACHE)

    todo: dict[str, str] = {}
    for key, text in zip(keys, texts):
        if key not in cache and key not in todo:
            todo[key] = text
    if cache.rows:
        logger.info("Embedding cache: %d/%d chunks already embedded", len(texts) - len(todo), len(texts))

    if todo:
        batch_size = max(1, config.EMBED_BATCH_SIZE)
        items = list(todo.items())
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        iterator = batches
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(batches, desc="Embedding", unit="batch")
            except ImportError:
                pass

        limiter = _RateLimiter(config.EMBED_RPM)
        for batch in iterator:
            vectors = _embed_with_retry(embeddings, [text for _, text in batch], limiter)
            cache.add([key for key, _ in batch], np.asarray(vectors, dtype=np.float32))

    index = faiss.IndexFlatL2(cache.dim)
    for i in range(0, len(keys), _ADD_BLOCK):
        index.add(cache.get(keys[i : i + _ADD_BLOCK]))

    ids = [str(uuid.uuid4()) for _ in chunks]
    docstore = InMemoryDocstore(
        {id_: Document(id=id_, page_content=c.page_content, metadata=c.metadata) for id_, c in zip(ids, chunks)}
    )
    # Same construction FAISS.from_embeddings performs (flat L2 index,
    # uuid-keyed docstore), minus its full float64 copy of every vector.
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=dict(enumerate(ids)),
    )


def save_index(
    store: FAISS,
    chunks: list[Document],
    index_dir: str | None = None,
    corpus: dict | None = None,
) -> Path:
    path = Path(index_dir or config.INDEX_DIR)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))

    with (path / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps({"text": chunk.page_content, "metadata": chunk.metadata}) + "\n")

    meta = {
        "embed_provider": config.EMBED_PROVIDER,
        "chat_provider": config.PROVIDER,
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "header_mode": config.HEADER_MODE,
        "chunk_count": len(chunks),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if corpus:
        meta["corpus"] = corpus
    (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
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
