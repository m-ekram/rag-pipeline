"""Dense retrieval over Qdrant, with a local fallback.

`qdrant_client` is imported only where it is used. It pulls in grpc, whose
compiled module Windows Application Control blocks on some machines, and a
module-level import made the whole pipeline unimportable there.
`make_dense_index` picks Qdrant when a server answers and the library loads,
and the local NumPy index (`retrieval.local_dense`) otherwise.
"""

from __future__ import annotations

import time
import os
import uuid
import logging
from typing import Any, Iterable, Optional, Sequence

from ingestion.documents import Chunk
from .embedder import Embedder
from .types import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# Stable namespace so the same chunk_id always maps to the same point id across
# rebuilds — otherwise re-ingesting duplicates every chunk instead of updating.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def point_id(chunk_id: str) -> str:
    """Map an arbitrary chunk id onto a Qdrant-legal point id.

    Qdrant accepts only uint64 or UUID point ids; our chunk ids look like
    `3::0`, so they are hashed into a deterministic UUIDv5 and the original is
    kept in the payload.
    """
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class DenseIndex:
    """Qdrant-backed vector index over chunks."""

    def __init__(
        self,
        collection_name: str,
        embedder: Optional[Embedder] = None,
        *,
        url: str = DEFAULT_URL,
        client: Optional[Any] = None,
    ):
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=url)
        self.client = client

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def count(self) -> int:
        return self.client.count(self.collection_name).count

    def recreate(self) -> None:
        """Drop and recreate the collection, sized from the embedding model."""
        from qdrant_client.models import (
            Distance,
            ScalarQuantization,
            ScalarQuantizationConfig,
            ScalarType,
            VectorParams,
        )

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            for _ in range(20):
                if not self.client.collection_exists(self.collection_name):
                    break
                time.sleep(0.05)
        create_kwargs = {
            "collection_name": self.collection_name,
            "vectors_config": VectorParams(
                size=self.embedder.dimension, distance=Distance.COSINE
            ),
        }
        try:
            self.client.create_collection(
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=False,  # Memory-mapped on disk to keep RAM low at scale
                    )
                ),
                **create_kwargs,
            )
        except Exception:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(**create_kwargs)

        # Create payload indexes for fast filtered lookups
        if hasattr(self.client, "create_payload_index"):
            try:
                for field_name in ["doc_id", "block_type", "parent_id"]:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema="keyword",
                    )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="page_num",
                    field_schema="integer",
                )
            except Exception as e:
                logger.debug("Payload index creation note: %s", e)

    def index(
        self,
        chunks: Iterable[Chunk],
        *,
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> int:
        """Embed and upsert chunks. Returns the number indexed."""
        from qdrant_client.models import PointStruct

        batch: list[Chunk] = []
        total = 0

        def flush(items: Sequence[Chunk]) -> int:
            if not items:
                return 0
            vectors = self.embedder.embed_passages(
                [c.text for c in items], show_progress=show_progress
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id(c.chunk_id),
                        vector=vector.tolist(),
                        payload=c.to_payload(),
                    )
                    for c, vector in zip(items, vectors)
                ],
                wait=True,
            )
            return len(items)

        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                total += flush(batch)
                logger.info("Indexed %d chunks...", total)
                batch = []
        total += flush(batch)
        logger.info("Indexed %d chunks total.", total)
        return total

    def search(
        self,
        query: str,
        limit: int = 10,
        filter: Optional[Any] = None,
    ) -> list[ScoredChunk]:
        vector = self.embedder.embed_query(query).tolist()
        # `client.search()` was removed in qdrant-client 1.x; query_points is the
        # replacement and returns a response object wrapping `.points`.
        query_kwargs = {
            "collection_name": self.collection_name,
            "query": vector,
            "limit": limit,
            "with_payload": True,
        }
        if filter is not None:
            query_kwargs["query_filter"] = filter
        response = self.client.query_points(**query_kwargs)
        return [
            ScoredChunk(
                chunk_id=(point.payload or {}).get("chunk_id", str(point.id)),
                score=float(point.score),
                rank=rank,
                chunk=_chunk_from_payload(point.payload),
            )
            for rank, point in enumerate(response.points, 1)
        ]

    def get_by_page(self, page_num: int, limit: int = 50, doc_id: Optional[str] = None) -> list[Chunk]:
        """Fetch chunks matching a specific page number (and document, if given)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [FieldCondition(key="page_num", match=MatchValue(value=page_num))]
        if doc_id is not None:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        chunks = []
        for p in points:
            c = _chunk_from_payload(p.payload)
            if c:
                chunks.append(c)
        return chunks


def _chunk_from_payload(payload: Optional[dict]) -> Optional[Chunk]:
    if not payload:
        return None
    fields = {f for f in Chunk.__dataclass_fields__}
    return Chunk(**{k: v for k, v in payload.items() if k in fields})


# One client per URL for the whole process. `None` records that Qdrant was not
# usable, so later indexes do not pay the probe again.
_QDRANT_CLIENTS: dict[str, Any] = {}


def _qdrant_client_for(url: str) -> Optional[Any]:
    """A shared Qdrant client for `url`, or None when Qdrant is unusable here."""
    if url not in _QDRANT_CLIENTS:
        client = None
        try:
            from qdrant_client import QdrantClient

            QdrantClient(url=url, timeout=2.0).get_collections()
            client = QdrantClient(url=url)
        except Exception as exc:  # blocked grpc import, server not running, ...
            logger.info("Qdrant unavailable at %s (%s); using the local vector index.", url, exc)
        _QDRANT_CLIENTS[url] = client
    return _QDRANT_CLIENTS[url]


def make_dense_index(collection_name: str, embedder: Optional[Embedder] = None, *,
                     url: Optional[str] = None):
    """Vector index for `collection_name`: a Qdrant server if usable, else local.

    `RAG_VECTOR_STORE=local` or `=qdrant` forces the choice.
    """
    choice = os.environ.get("RAG_VECTOR_STORE", "auto").lower()
    url = url or DEFAULT_URL
    if choice != "local":
        client = _qdrant_client_for(url)
        if client is not None:
            return DenseIndex(collection_name, embedder, client=client)
        if choice == "qdrant":
            raise RuntimeError(f"RAG_VECTOR_STORE=qdrant, but Qdrant is not usable at {url}")
    from .local_dense import LocalDenseIndex

    return LocalDenseIndex(collection_name, embedder)
