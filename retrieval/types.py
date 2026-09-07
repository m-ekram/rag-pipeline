"""Shared retrieval result type."""

from dataclasses import dataclass
from typing import Optional

from ingestion.documents import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    """One retrieved chunk with the score that retrieved it.

    `score` is only comparable within a single retriever's result list — BM25
    scores and cosine similarities live on different scales, which is exactly
    why fusion is rank-based (RRF) rather than score-based.
    """

    chunk_id: str
    score: float
    rank: int = 0
    chunk: Optional[Chunk] = None

    @property
    def text(self) -> str:
        return self.chunk.text if self.chunk else ""
