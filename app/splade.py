"""Learned sparse retrieval (SPLADE) as the lexical leg of hybrid search.

BM25 only matches words that literally appear in both question and passage.
SPLADE runs each text through a masked-language model and emits weights over
the whole vocabulary, including terms the text implies but never uses - a
page about serving static files can carry weight on "css" or "images", a
question about checking that endpoints work can carry weight on "test".
Scoring is still a sparse dot product over an inverted index, so it slots in
exactly where BM25 did (`config.SPARSE = "splade"`).

Cost: encoding a chunk is a transformer pass, comparable to dense embedding at
ingest. Document vectors are cached on disk by text hash, like the dense cache.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

import config
from app.doc2query import indexed_text

logger = logging.getLogger(__name__)

SparseVector = tuple[np.ndarray, np.ndarray]  # (vocabulary indices, weights)


class SparseEncoder:
    """A fastembed SPLADE model plus an append-only cache of document vectors."""

    def __init__(self, model_name: str):
        from fastembed import SparseTextEmbedding

        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir=config.MODEL_CACHE_DIR)
        slug = re.sub(r"[^\w.-]", "_", model_name)
        self._cache_path = Path(config.EMBED_CACHE_DIR) / f"{slug}.sparse.jsonl"

    def encode_query(self, text: str) -> SparseVector:
        vector = next(iter(self._model.query_embed(text)))
        return np.asarray(vector.indices), np.asarray(vector.values)

    def _load(self) -> dict[str, SparseVector]:
        cache: dict[str, SparseVector] = {}
        if not self._cache_path.exists():
            return cache
        with self._cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                    cache[record["k"]] = (np.asarray(record["i"], dtype=np.int64), np.asarray(record["v"], dtype=np.float32))
                except (json.JSONDecodeError, KeyError):
                    continue  # a torn final line from an interrupted run
        return cache

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        cache = self._load() if config.EMBED_CACHE else {}
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cache]
        if todo:
            logger.info("SPLADE: encoding %d/%d chunks", len(todo), len(texts))
            fh = None
            if config.EMBED_CACHE:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                fh = self._cache_path.open("a", encoding="utf-8")
            try:
                for start in range(0, len(todo), 64):
                    batch = todo[start : start + 64]
                    for (key, _), vector in zip(batch, self._model.embed([t for _, t in batch], batch_size=32)):
                        indices = np.asarray(vector.indices, dtype=np.int64)
                        values = np.asarray(vector.values, dtype=np.float32)
                        cache[key] = (indices, values)
                        if fh:
                            fh.write(json.dumps({"k": key, "i": indices.tolist(), "v": values.tolist()}) + "\n")
                    if fh:
                        fh.flush()  # survive a kill between batches
            finally:
                if fh:
                    fh.close()
        return [cache[k] for k in keys]


@functools.lru_cache(maxsize=2)
def get_sparse_encoder(model_name: str | None = None) -> SparseEncoder:
    return SparseEncoder(model_name or config.SPARSE_MODEL)


class SpladeRetriever(BaseRetriever):
    """Drop-in for the BM25 leg: same fields callers use (`docs`, `k`)."""

    docs: list[Document] = Field(repr=False)
    k: int = 4
    postings: dict[int, tuple[np.ndarray, np.ndarray]] = Field(repr=False)
    encoder: Any = Field(repr=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, documents: Iterable[Document], encoder: Any = None, **kwargs) -> "SpladeRetriever":
        docs = list(documents)
        encoder = encoder or get_sparse_encoder()
        # Chunk text plus any doc2query expansion; the returned documents keep
        # their original page_content.
        return cls.from_vectors(docs, encoder.encode_documents([indexed_text(d) for d in docs]), encoder, **kwargs)

    @classmethod
    def from_vectors(cls, docs: list[Document], vectors: list[SparseVector], encoder: Any, **kwargs) -> "SpladeRetriever":
        doc_ids: dict[int, list[int]] = {}
        weights: dict[int, list[float]] = {}
        for i, (indices, values) in enumerate(vectors):
            for term, weight in zip(np.asarray(indices).tolist(), np.asarray(values).tolist()):
                doc_ids.setdefault(term, []).append(i)
                weights.setdefault(term, []).append(weight)
        postings = {
            term: (np.array(ids, dtype=np.int64), np.array(weights[term], dtype=np.float32)) for term, ids in doc_ids.items()
        }
        return cls(docs=docs, postings=postings, encoder=encoder, **kwargs)

    def scores(self, query: str) -> np.ndarray:
        total = np.zeros(len(self.docs), dtype=np.float32)
        indices, values = self.encoder.encode_query(query)
        for term, weight in zip(np.asarray(indices).tolist(), np.asarray(values).tolist()):
            posting = self.postings.get(term)
            if posting is not None:
                ids, doc_weights = posting
                total[ids] += weight * doc_weights
        return total

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        top = np.argsort(self.scores(query))[::-1][: self.k]
        return [self.docs[i] for i in top]
