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

    for results in (dense_results, lexical_results):
        # Rank comes from list position, never from item.rank. ScoredChunk.rank
        # defaults to 0, so a retriever that forgets to set it would give every
        # result 1/(k+0) — identical scores, and the ranking silently collapses
        # to insertion order.
        for position, item in enumerate(results, 1):
            cid = item.chunk_id
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + position))
            # Keep whichever copy actually carries the chunk payload: a dense hit
            # whose Qdrant payload was missing would otherwise overwrite a
            # lexical hit that has the text, and the evidence vanishes downstream.
            if cid not in chunk_map or chunk_map[cid].chunk is None:
                chunk_map[cid] = item

    # Sort descending by fused RRF score
    # Tie-break on chunk_id so fusion output is deterministic across runs.
    sorted_items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

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
