"""Dependency-free dense index: a normalised vector matrix on local disk.

Used whenever Qdrant is not usable. Its client imports grpc, whose compiled
module Windows Application Control blocks on some machines, and embedded Qdrant
locks its storage folder per process. For one user's corpus (tens to a few
hundred thousand chunks) a single matrix-vector product is also faster than
either, and the index persists, so re-opening a folder needs no re-embedding.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ingestion.documents import Chunk
from .embedder import Embedder
from .types import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / ".cache" / "vectors"
_CHUNK_FIELDS = frozenset(Chunk.__dataclass_fields__)


def _chunk_from_payload(payload: dict) -> Chunk:
    return Chunk(**{k: v for k, v in payload.items() if k in _CHUNK_FIELDS})


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalDenseIndex:
    """Vector index stored as `vectors.npy` (float16) plus `payloads.jsonl`.

    Vectors are held in RAM as float32 for search and written as float16,
    halving disk use; cosine similarity between unit vectors loses nothing
    measurable at that precision.
    """

    def __init__(
        self,
        collection_name: str,
        embedder: Optional[Embedder] = None,
        *,
        root: Path | str = DEFAULT_ROOT,
    ):
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        self.dir = Path(root) / collection_name
        self._vectors: Optional[np.ndarray] = None
        self._payloads: list[dict] = []
        self._pages: Optional[dict[int, list[int]]] = None
        self._load()

    # -- storage ---------------------------------------------------------

    @property
    def _vector_path(self) -> Path:
        return self.dir / "vectors.npy"

    @property
    def _payload_path(self) -> Path:
        return self.dir / "payloads.jsonl"

    def _load(self) -> None:
        if not (self._vector_path.exists() and self._payload_path.exists()):
            return
        try:
            vectors = np.load(self._vector_path).astype(np.float32)
            with open(self._payload_path, encoding="utf-8") as handle:
                payloads = [json.loads(line) for line in handle if line.strip()]
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable vector index %s: %s", self.dir, exc)
            return
        if len(payloads) != len(vectors):
            # Two files are replaced one after the other; a crash between the
            # replaces leaves them disagreeing, which must read as "no index".
            logger.warning("Ignoring inconsistent vector index %s (%d vectors, %d payloads)",
                           self.dir, len(vectors), len(payloads))
            return
        self._vectors, self._payloads = vectors, payloads

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so a crash mid-write cannot leave a truncated file
        # that a later run would load as a valid index.
        tmp_vectors = self.dir / "vectors.tmp.npy"
        np.save(tmp_vectors, self._vectors.astype(np.float16))
        tmp_payloads = self.dir / "payloads.tmp.jsonl"
        with open(tmp_payloads, "w", encoding="utf-8") as handle:
            for payload in self._payloads:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        os.replace(tmp_vectors, self._vector_path)
        os.replace(tmp_payloads, self._payload_path)

    # -- the DenseIndex interface ----------------------------------------

    def exists(self) -> bool:
        return self._vectors is not None

    def count(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def recreate(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
        self._vectors, self._payloads, self._pages = None, [], None

    def index(
        self,
        chunks: Iterable[Chunk],
        *,
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> int:
        """Embed and append chunks, then persist. Call `recreate()` first to replace."""
        items = list(chunks)
        if not items:
            return 0
        blocks = []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            vectors = self.embedder.embed_passages([c.text for c in batch], show_progress=show_progress)
            blocks.append(np.asarray(vectors, dtype=np.float32))
            logger.info("Embedded %d/%d chunks", start + len(batch), len(items))
        new = _normalise(np.vstack(blocks))
        self._vectors = new if self._vectors is None else np.vstack([self._vectors, new])
        self._payloads.extend(c.to_payload() for c in items)
        self._pages = None
        self._save()
        return len(items)

    def search(self, query: str, limit: int = 10, filter=None) -> list[ScoredChunk]:
        if filter is not None:
            raise NotImplementedError("LocalDenseIndex does not support Qdrant filters")
        if self._vectors is None or not len(self._vectors) or limit <= 0:
            return []
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm:
            q = q / norm
        scores = self._vectors @ q
        k = min(limit, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            ScoredChunk(
                chunk_id=self._payloads[i].get("chunk_id", str(i)),
                score=float(scores[i]),
                rank=rank,
                chunk=_chunk_from_payload(self._payloads[i]),
            )
            for rank, i in enumerate(top, 1)
        ]

    def get_by_page(self, page_num: int, limit: int = 50, doc_id: Optional[str] = None) -> list[Chunk]:
        """Chunks on one page, optionally of one document, in index order."""
        if self._pages is None:
            self._pages = {}
            for i, payload in enumerate(self._payloads):
                page = payload.get("page_num")
                if page is not None:
                    self._pages.setdefault(int(page), []).append(i)
        hits = [
            _chunk_from_payload(self._payloads[i])
            for i in self._pages.get(int(page_num), [])
            if doc_id is None or self._payloads[i].get("doc_id") == doc_id
        ]
        return hits[:limit]
