"""Reciprocal Rank Fusion (RRF) for hybrid retrieval merging.

RRF(d) = sum(1 / (k + rank(d))) across candidate rankings.
Provides scale-invariant rank merging without manual score normalization.
"""

from __future__ import annotations

from typing import Sequence
from .types import ScoredChunk


def reciprocal_rank_fusion(
    dense_results: Sequence[ScoredChunk],
    lexical_results: Sequence[ScoredChunk],
    *,
    k: int = 60,
    limit: int = 15,
) -> list[ScoredChunk]:
    """Merge dense and lexical rankings using Reciprocal Rank Fusion.

    Parameters:
        dense_results: Scored chunks ranked by vector similarity (rank 1..N).
        lexical_results: Scored chunks ranked by BM25/FTS5 (rank 1..M).
        k: Smoothing constant (default: 60, per standard information retrieval literature).
        limit: Max merged results to return.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, ScoredChunk] = {}

    for item in dense_results:
        cid = item.chunk_id
        chunk_map[cid] = item
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + item.rank))

    for item in lexical_results:
        cid = item.chunk_id
        if cid not in chunk_map:
            chunk_map[cid] = item
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + item.rank))

    # Sort descending by fused RRF score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    merged: list[ScoredChunk] = []
    for rank, (cid, fused_score) in enumerate(sorted_items, 1):
        orig = chunk_map[cid]
        merged.append(
            ScoredChunk(
                chunk_id=cid,
                score=fused_score,
                rank=rank,
                chunk=orig.chunk,
            )
        )

    return merged
