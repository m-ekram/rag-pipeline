"""Dense retrieval with optional cross-encoder reranking.

Pipeline:
    Dense retrieval
        ↓
    Candidate set
        ↓
    Cross-Encoder reranker
        ↓
    Final results

BM25 and RRF are intentionally not used in the production retrieval path.
"""

import logging
from typing import Optional

from .dense import DenseIndex
from .reranker import CrossEncoderReranker
from .types import ScoredChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense retrieval with optional cross-encoder reranking.

    The class name is retained for compatibility with the existing
    evaluation code and imports. The actual retrieval pipeline is
    dense-only.
    """

    def __init__(
        self,
        dense: DenseIndex,
        *,
        candidate_limit: int = 50,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        if dense is None:
            raise ValueError(
                "HybridRetriever requires a dense retriever"
            )

        if candidate_limit <= 0:
            raise ValueError(
                "candidate_limit must be positive"
            )

        self.dense = dense
        self.candidate_limit = candidate_limit
        self.reranker = reranker

    @property
    def mode(self) -> str:
        """Return the active retrieval pipeline variant."""

        if self.reranker is not None:
            return "dense+rerank"

        return "dense"

    def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ScoredChunk]:
        """Retrieve dense candidates and optionally rerank them.

        Retrieval flow:

        1. Retrieve up to ``candidate_limit`` chunks using dense retrieval.
        2. If a reranker is configured, cross-encode those candidates.
        3. Return the requested number of final results.
        """

        if limit <= 0:
            return []

        candidates = self.dense.search(
            query,
            limit=self.candidate_limit,
        )

        logger.debug(
            "Dense retrieval returned %d candidates",
            len(candidates),
        )

        if self.reranker is not None:
            logger.debug(
                "Reranking %d candidates for query",
                len(candidates),
            )

            return self.reranker.rerank(
                query,
                candidates,
                limit=limit,
            )

        return candidates[:limit]