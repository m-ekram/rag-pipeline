"""Lexical-first retrieval that preserves BM25's abstention.

Measured on the 144-page Devanagari electoral corpus (600 chunks):

    BM25   answerable min 1.587   nonsense 0/6 returned   -> categorical abstention
    DENSE  answerable min 0.825   nonsense 6/6 returned   -> margin +0.0272

Two consequences drive this design.

**Plain RRF would destroy abstention.** Dense returns something for every query,
so fusing the two lists unconditionally means the fused list is never empty —
the system loses the sharpest and cheapest abstention signal it has, and gains a
threshold that must be calibrated inside a 0.027-wide window. Fusion therefore
runs only when BM25 has already found something.

**Dense rescue is available but partial, and off by default.** BM25 cannot
bridge scripts: "list of voters and their names" returns nothing against
Devanagari pages, where dense retrieves correctly. But the ranges overlap —
measured on the 1,823-chunk record-chunked corpus:

    cross-lingual   0.7815  Where is the polling station?      <- below nonsense
                    0.8279  voter identity card number
                    0.8383  list of voters and their names
                    0.8817  electoral roll 2025 Bihar
    nonsense        0.7923  रोटी कैसे बनाएं                     <- above one real query
                    0.7083 – 0.7658  (the other five)

No floor admits all four cross-lingual queries while excluding all six nonsense
ones. A floor at 0.80 is the best available trade: **3 of 4 cross-lingual
queries recovered, 0 of 6 nonsense admitted.** That is the default, and rescue
stays off unless asked for, so abstention is never lost by accident.

These are measurements on 10 queries, not a calibrated threshold.
"""

import logging
from typing import Optional

from .bm25 import BM25Index
from .dense import DenseIndex
from .fusion import DEFAULT_K, reciprocal_rank_fusion
from .types import ScoredChunk

logger = logging.getLogger(__name__)


class LexicalFirstRetriever:
    """BM25 leads; dense refines the ranking or rescues cross-lingual queries."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: Optional[DenseIndex] = None,
        *,
        candidate_limit: int = 50,
        rrf_k: int = DEFAULT_K,
        weights: Optional[tuple[float, float]] = None,
        dense_rescue: bool = False,
        rescue_min_score: float = 0.80,
        reranker=None,
    ):
        if bm25 is None:
            raise ValueError("LexicalFirstRetriever requires a BM25 index")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")

        self.bm25 = bm25
        self.dense = dense
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k
        # Default equal weighting; BM25 already leads by running first.
        self.weights = weights
        self.dense_rescue = dense_rescue
        # 0.80 sits above the highest-scoring nonsense query (0.7923) and below
        # three of the four cross-lingual queries. Measured, not calibrated —
        # Phase 3 replaces it with a risk-coverage-derived threshold.
        self.rescue_min_score = rescue_min_score
        self.reranker = reranker

    @property
    def mode(self) -> str:
        parts = ["bm25"]
        if self.dense is not None:
            parts.append("rrf" if not self.dense_rescue else "rrf+rescue")
        if self.reranker is not None:
            parts.append("rerank")
        return "+".join(parts)

    def retrieve(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        if limit <= 0:
            return []

        lexical = self.bm25.search(query, limit=self.candidate_limit)

        if not lexical:
            # BM25 found no matching term. This is the abstention signal — do not
            # paper over it with dense results unless rescue is explicitly on.
            if not (self.dense_rescue and self.dense is not None):
                logger.debug("No lexical match for %r; abstaining.", query)
                return []

            rescued = self.dense.search(query, limit=self.candidate_limit)
            rescued = [hit for hit in rescued if hit.score >= self.rescue_min_score]
            if not rescued:
                return []
            logger.debug("Lexical miss rescued by dense for %r (%d hits).",
                         query, len(rescued))
            return self._finish(query, rescued, limit)

        if self.dense is None:
            return self._finish(query, lexical, limit)

        semantic = self.dense.search(query, limit=self.candidate_limit)
        fused = reciprocal_rank_fusion(
            [lexical, semantic], k=self.rrf_k, weights=self.weights
        )
        return self._finish(query, fused, limit)

    def _finish(self, query: str, candidates: list[ScoredChunk],
                limit: int) -> list[ScoredChunk]:
        if self.reranker is not None and candidates:
            return self.reranker.rerank(query, candidates, limit=limit)
        return candidates[:limit]
