"""BM25 over an inverted index.

rank_bm25, which LangChain's BM25Retriever wraps, scores a query by walking
every document's term dict in Python, once per query term. At 48k chunks (a
12,000-page corpus) that was ~400 ms per query: the single largest cost of
serving a large corpus.

This keeps rank_bm25's exact BM25Okapi scoring (k1, b, the epsilon idf floor)
and its tie-breaking, but stores one posting list per term with the
per-document weight precomputed. A query touches only the documents that
contain its terms, and the results are identical - tests/test_pipeline.py
checks that against rank_bm25 directly.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


def default_preprocess(text: str) -> list[str]:
    """Same tokenisation as LangChain's BM25Retriever default."""
    return text.split()


class FastBM25Retriever(BaseRetriever):
    """Drop-in for BM25Retriever: same fields callers use (`docs`, `k`)."""

    docs: list[Document] = Field(repr=False)
    k: int = 4
    postings: dict[str, tuple[np.ndarray, np.ndarray]] = Field(repr=False)
    preprocess_func: Callable[[str], list[str]] = default_preprocess

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        preprocess_func: Callable[[str], list[str]] = default_preprocess,
        **kwargs,
    ) -> "FastBM25Retriever":
        docs = list(documents)
        if not docs:
            raise ValueError("BM25 needs at least one document")

        doc_ids: dict[str, list[int]] = {}
        term_freqs: dict[str, list[int]] = {}
        lengths = []
        for i, doc in enumerate(docs):
            tokens = preprocess_func(doc.page_content)
            lengths.append(len(tokens))
            # Counter preserves first-occurrence order, as rank_bm25's dicts do;
            # that order feeds the idf average, so matching it keeps floats identical.
            for term, tf in Counter(tokens).items():
                doc_ids.setdefault(term, []).append(i)
                term_freqs.setdefault(term, []).append(tf)

        n = len(docs)
        doc_len = np.array(lengths)
        avgdl = sum(lengths) / n

        idf: dict[str, float] = {}
        idf_sum = 0.0
        negative = []
        for term, ids in doc_ids.items():
            value = math.log(n - len(ids) + 0.5) - math.log(len(ids) + 0.5)
            idf[term] = value
            idf_sum += value
            if value < 0:
                negative.append(term)
        eps = epsilon * (idf_sum / len(idf))
        for term in negative:
            idf[term] = eps

        postings = {}
        for term, ids in doc_ids.items():
            ids_arr = np.array(ids, dtype=np.int64)
            tf = np.array(term_freqs[term])
            weight = idf[term] * (tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[ids_arr] / avgdl)))
            postings[term] = (ids_arr, weight)

        return cls(docs=docs, postings=postings, preprocess_func=preprocess_func, **kwargs)

    def scores(self, query: str) -> np.ndarray:
        total = np.zeros(len(self.docs))
        for term in self.preprocess_func(query):
            posting = self.postings.get(term)
            if posting is not None:
                ids, weight = posting
                total[ids] += weight
        return total

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        top = np.argsort(self.scores(query))[::-1][: self.k]
        return [self.docs[i] for i in top]
