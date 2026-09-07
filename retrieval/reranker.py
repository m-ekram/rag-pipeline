"""Cross-encoder reranking over the fused candidate set.

The reranker is optional and runs locally using sentence-transformers.
It operates only on the candidate set returned by BM25 + dense retrieval,
so it does not affect the cost of indexing or retrieval when disabled.

The normalised score is also suitable for the later abstention gate.
"""

import logging
import math
from typing import Optional, Sequence

from .types import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _sigmoid(x: float) -> float:
    """Convert a raw cross-encoder logit into a value in (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))

    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class CrossEncoderReranker:
    """Lazily-loaded local cross-encoder reranker."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        normalise: bool = True,
        device: Optional[str] = None,
        model=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalise = normalise
        self.device = device

        # Lazy loading:
        # the model is NOT loaded when HybridRetriever is imported.
        self._model = model

    @property
    def model(self):
        """Load the cross-encoder only when reranking is actually requested."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading cross-encoder %s...",
                self.model_name,
            )

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )

        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        limit: Optional[int] = None,
    ) -> list[ScoredChunk]:
        """Rerank retrieved candidates against the query."""

        scorable = [
            candidate
            for candidate in candidates
            if candidate.chunk is not None
        ]

        if len(scorable) != len(candidates):
            logger.warning(
                "Dropped %d candidates with no chunk payload",
                len(candidates) - len(scorable),
            )

        if not scorable:
            return []

        pairs = [
            (query, candidate.chunk.text)
            for candidate in scorable
        ]

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
        )

        scores = [
            _sigmoid(float(score)) if self.normalise else float(score)
            for score in raw_scores
        ]

        order = sorted(
            range(len(scorable)),
            key=lambda i: scores[i],
            reverse=True,
        )

        if limit is not None:
            order = order[:limit]

        return [
            ScoredChunk(
                chunk_id=scorable[i].chunk_id,
                score=scores[i],
                rank=rank,
                chunk=scorable[i].chunk,
            )
            for rank, i in enumerate(order, 1)
        ]