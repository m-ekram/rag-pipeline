"""Cross-encoder reranking over the fused candidate set.

Runs locally (sentence-transformers CrossEncoder) — no API involved. On this
project the reranked score is not just a ranking signal: it is the quantity the
abstention gate thresholds on, so its calibration is load-bearing.

ms-marco-MiniLM outputs raw logits, roughly -11..+11, *not* probabilities.
Thresholding raw logits works but the numbers are unintuitive and shift between
reranker checkpoints, so `normalise=True` maps them through a sigmoid into
(0, 1). Fixing the scale now means a calibrated threshold stays meaningful if
the reranker is swapped for bge-reranker-base later (a Stretch item).
"""

import functools
import logging
from typing import Optional, Sequence

from retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MULTILINGUAL_LIGHT = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MULTILINGUAL_BASE = "BAAI/bge-reranker-base"

# Query + passage tokens scored per pair. Attention cost grows with the square
# of the length, so an uncapped 512-token pair costs ~1.8x a 384-token one on
# CPU; child chunks are ~200 words, so 384 rarely truncates anything.
DEFAULT_MAX_LENGTH = 384


def _sigmoid(x: float) -> float:
    import math

    # Guard against overflow on the large-magnitude logits the model can emit.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class CrossEncoderReranker:
    """Lazily-loaded cross-encoder. Importing this module downloads nothing."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        normalise: bool = True,
        device: Optional[str] = None,
        max_length: Optional[int] = DEFAULT_MAX_LENGTH,
        model=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalise = normalise
        self.device = device
        self.max_length = max_length
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s...", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device,
                                       max_length=self.max_length)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        limit: Optional[int] = None,
    ) -> list[ScoredChunk]:
        """Re-score candidates against the query and return them re-ordered.

        Candidates without their chunk payload cannot be scored — the reranker
        needs the text, not just the id — so they are dropped rather than
        silently scored as zero.
        """
        scorable = [c for c in candidates if c.chunk is not None]
        if len(scorable) != len(candidates):
            logger.warning("Dropped %d candidates with no chunk payload",
                           len(candidates) - len(scorable))
        if not scorable:
            return []

        pairs = [(query, c.chunk.text) for c in scorable]
        raw = self.model.predict(pairs, batch_size=self.batch_size)
        scores = [_sigmoid(float(s)) if self.normalise else float(s) for s in raw]

        order = sorted(range(len(scorable)), key=lambda i: scores[i], reverse=True)
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


@functools.lru_cache(maxsize=4)
def get_reranker(model_name: str = DEFAULT_MODEL) -> CrossEncoderReranker:
    """One reranker per model for the whole process; loading weights costs seconds."""
    return CrossEncoderReranker(model_name)
