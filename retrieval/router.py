"""Query Understanding, Intent Routing, and Hybrid Retrieval Execution.

Routes queries based on intent:
- Physical Page Lookup ("What is on page 66?"): Direct payload filter bypassing embedding drift.
- Exact Entity / ID ("BR/35/207/291052", "SHS1291525"): Exact lexical matching with FTS5.
- Thematic / Semantic ("Proposed land use percentage in PPA"): Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import enum
import logging
import re
from typing import Optional, Protocol, Sequence

from ingestion.documents import Chunk
from .fts5_index import FTS5Index
from .rrf import reciprocal_rank_fusion
from .types import ScoredChunk

logger = logging.getLogger(__name__)

# Regex for detecting explicit page lookups: "page 66", "on page 12", "page no 3"
_PAGE_PATTERN = re.compile(r"(?:on\s+page|page\s+no\.?|page\s+number|\bp\.?)\s*(\d+)", re.IGNORECASE)

# Regex for detecting structured IDs: "BR/35/207/291052", "SHS5361415", "JDK2924306"
_ID_PATTERN = re.compile(r"\b([A-Z]{2,4}/\d+/\d+/\d+|[A-Z]{3}\d{7})\b", re.IGNORECASE)


class QueryIntent(enum.Enum):
    PAGE_LOOKUP = "page_lookup"
    EXACT_ENTITY = "exact_entity"
    HYBRID_SEMANTIC = "hybrid_semantic"


class IntentRouter:
    """Unified Query Intent Router and Hybrid Retriever."""

    def __init__(
        self,
        lexical: FTS5Index,
        dense,  # DenseIndex
        reranker=None,  # CrossEncoderReranker
        *,
        candidate_limit: int = 15,
    ):
        self.lexical = lexical
        self.dense = dense
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    def classify(self, query: str) -> tuple[QueryIntent, dict]:
        """Classify user intent and extract query parameters."""
        page_match = _PAGE_PATTERN.search(query)
        if page_match:
            return QueryIntent.PAGE_LOOKUP, {"page_num": int(page_match.group(1))}

        id_match = _ID_PATTERN.search(query)
        if id_match:
            return QueryIntent.EXACT_ENTITY, {"entity_id": id_match.group(1)}

        return QueryIntent.HYBRID_SEMANTIC, {}

    def retrieve(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        """Execute retrieval based on classified query intent."""
        intent, params = self.classify(query)

        candidates: list[ScoredChunk] = []

        if intent == QueryIntent.PAGE_LOOKUP:
            page_num = params["page_num"]
            logger.info("Routing query to PAGE_LOOKUP for page %d", page_num)
            page_chunks = self.lexical.get_by_page(page_num)
            if not page_chunks and hasattr(self.dense, "get_by_page"):
                page_chunks = self.dense.get_by_page(page_num)

            candidates = [
                ScoredChunk(chunk_id=c.chunk_id, score=1.0 - (idx * 0.01), rank=idx + 1, chunk=c)
                for idx, c in enumerate(page_chunks[: self.candidate_limit])
            ]

        elif intent == QueryIntent.EXACT_ENTITY:
            entity_id = params["entity_id"]
            logger.info("Routing query to EXACT_ENTITY for ID '%s'", entity_id)
            lexical_hits = self.lexical.search(entity_id, limit=self.candidate_limit)
            if not lexical_hits and len(entity_id) >= 6:
                prefix_query = entity_id[:-1] + "*"
                logger.info("Exact ID miss. Trying prefix search: '%s'", prefix_query)
                lexical_hits = self.lexical.search(prefix_query, limit=self.candidate_limit)
            if not lexical_hits and hasattr(self.lexical, "fuzzy_search_epic"):
                logger.info("Exact and prefix miss. Trying Levenshtein fuzzy search for: '%s'", entity_id)
                lexical_hits = self.lexical.fuzzy_search_epic(entity_id, max_distance=2, limit=self.candidate_limit)
            if lexical_hits:
                candidates = lexical_hits
            else:
                # If exact ID didn't hit in lexical, fall back to hybrid search
                intent = QueryIntent.HYBRID_SEMANTIC

        if intent == QueryIntent.HYBRID_SEMANTIC or not candidates:
            logger.info("Routing query to HYBRID_SEMANTIC (RRF k=60)")
            dense_hits = self.dense.search(query, limit=self.candidate_limit)
            lexical_hits = self.lexical.search(query, limit=self.candidate_limit)

            candidates = reciprocal_rank_fusion(
                dense_hits, lexical_hits, k=60, limit=self.candidate_limit
            )

        # Cross-encoder neural reranking if configured
        if self.reranker and candidates:
            reranked = self.reranker.rerank(query, candidates, limit=limit)
            return reranked

        return candidates[:limit]
