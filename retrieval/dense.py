"""Dense retrieval over Qdrant."""

import time
import os
import uuid
import logging
from typing import Iterable, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

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
        client: Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        self.client = client or QdrantClient(url=url)

    def recreate(self) -> None:
        """Drop and recreate the collection, sized from the embedding model."""
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
        filter: Optional[Filter] = None,
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

    def get_by_page(self, page_num: int, limit: int = 50) -> list[Chunk]:
        """Fetch chunks matching a specific page number."""
        page_filter = Filter(
            must=[FieldCondition(key="page_num", match=MatchValue(value=page_num))]
        )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=page_filter,
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
