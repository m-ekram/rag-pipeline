"""Rank fusion.

RRF is rank-based on purpose: BM25 scores and cosine similarities are not
comparable, so any weighted-sum over raw scores needs per-corpus normalisation
that silently shifts when the corpus changes — which is exactly what the
contamination sweep does. Rank-based fusion is invariant to that.
"""

from typing import Iterable, Optional, Sequence

from .types import ScoredChunk

DEFAULT_K = 60  # Cormack et al. 2009; large k flattens the contribution of top ranks


def reciprocal_rank_fusion(
    result_lists: Sequence[Iterable[ScoredChunk]],
    *,
    k: int = DEFAULT_K,
    weights: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists by summing 1 / (k + rank).

    `weights` supports the Stretch "weighted fusion" comparison without a second
    implementation; it defaults to equal weighting.
    """
    lists = [list(r) for r in result_lists]
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights must have one entry per result list")

    scores: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}

    for weight, results in zip(weights, lists):
        for position, result in enumerate(results, 1):
            # Trust list order rather than a possibly-unset .rank field.
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (k + position)
            # Keep whichever copy actually carries the chunk payload.
            if result.chunk_id not in chunks or chunks[result.chunk_id].chunk is None:
                chunks[result.chunk_id] = result

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        ordered = ordered[:limit]

    return [
        ScoredChunk(chunk_id=cid, score=score, rank=rank, chunk=chunks[cid].chunk)
        for rank, (cid, score) in enumerate(ordered, 1)
    ]
