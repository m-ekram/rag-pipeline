"""Dense embedding wrapper around bge-small-en-v1.5.

Two details here are easy to get wrong and both silently degrade recall:

1. BGE retrieval models expect an *instruction prefix on the query only*, not on
   passages. Embedding queries without it costs measurable nDCG.
2. Vectors must be L2-normalised for cosine distance to behave; Qdrant's COSINE
   metric normalises internally, but the BM25/dense score-fusion path and any
   local similarity maths assume unit vectors.
"""

import functools
import logging
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# The prefix the BGE authors specify for retrieval queries.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# bge-small-en-v1.5 is English-only. On Devanagari its tokeniser produces no
# [UNK]s — which makes the failure silent — but it shatters every word into
# single characters ("निर्वाचक" -> न ##ि ##र ##व ...), so the vectors carry
# almost no meaning. Hindi corpora need a multilingual model.
#
# multilingual-e5-small is the recommended swap here: it is also 384-dimensional,
# so an existing Qdrant collection's vector size does not change.
MULTILINGUAL_MODEL = "intfloat/multilingual-e5-small"

# Each family wants its own prefixes; using the wrong ones costs real recall.
# (query_prefix, passage_prefix)
_PREFIXES: dict[str, tuple[str, str]] = {
    "bge-small-en": (QUERY_INSTRUCTION, ""),
    "bge-base-en": (QUERY_INSTRUCTION, ""),
    "bge-large-en": (QUERY_INSTRUCTION, ""),
    "multilingual-e5": ("query: ", "passage: "),
    "e5-small": ("query: ", "passage: "),
    "e5-base": ("query: ", "passage: "),
    "e5-large": ("query: ", "passage: "),
    "bge-m3": ("", ""),
    "paraphrase-multilingual": ("", ""),
}


def prefixes_for(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) for a model, matched by substring."""
    lowered = model_name.lower()
    for marker, pair in _PREFIXES.items():
        if marker in lowered:
            return pair
    return ("", "")


class Embedder:
    """Lazily-loaded sentence-transformers wrapper.

    The model is loaded on first use so importing this module stays cheap —
    tests and the API's import path should not pull ~130MB of weights.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        query_instruction: Optional[str] = None,
        passage_instruction: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        default_query, default_passage = prefixes_for(model_name)
        # Explicit arguments win; otherwise the model's own convention is used.
        self.query_instruction = (
            default_query if query_instruction is None else query_instruction
        )
        self.passage_instruction = (
            default_passage if passage_instruction is None else passage_instruction
        )
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s...", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        """Vector size, read from the model rather than hardcoded."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: Iterable[str], *, show_progress: bool = True) -> np.ndarray:
        """Embed documents/chunks, applying the model's passage prefix if it has one."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self.passage_instruction:
            texts = [f"{self.passage_instruction}{t}" for t in texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    # Alias for document embedding compatibility
    embed_documents = embed_passages

    def embed_queries(self, texts: Iterable[str], *, show_progress: bool = False) -> np.ndarray:
        """Embed queries, applying the BGE retrieval instruction prefix."""
        texts = [f"{self.query_instruction}{t}" for t in texts]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]


@functools.lru_cache(maxsize=4)
def get_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    """One Embedder per model for the whole process.

    Loading the weights costs seconds (far more on a cold disk), and the API
    used to build a fresh Embedder, and reload the model, on every index.
    """
    return Embedder(model_name)
