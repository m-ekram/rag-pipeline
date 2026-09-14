"""Retrieval.

First stage: dense search (FAISS) fused with BM25. Dense search can smooth
exact identifiers - error codes, CLI flags, config keys - away; BM25 nails
those and misses paraphrases. Rank fusion keeps both.

MMR sits on the dense leg: it pulls FETCH_K candidates and picks ones that are
relevant *and* mutually dissimilar. On the FastAPI corpus that hurt (see
config.MMR_LAMBDA), so lambda defaults to 1.0 - plain relevance.

Optional second stage (RERANK=true): a cross-encoder reads each (question,
passage) pair jointly and re-scores the first stage's RERANK_FETCH_K
candidates, keeping TOP_K. Bi-encoder retrieval compares two independently
computed vectors; a cross-encoder sees both texts at once, which is what
recovers paraphrased questions whose wording shares little with the source.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

import config
from app.bm25 import FastBM25Retriever
from app.providers import get_cross_encoder

logger = logging.getLogger(__name__)


def _legacy(name: str):
    """langchain >= 1.0 moved the classic retrievers into langchain-classic."""
    import importlib

    for module in ("langchain_classic.retrievers", "langchain.retrievers"):
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
    for module in ("langchain_classic.retrievers.document_compressors", "langchain.retrievers.document_compressors"):
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            continue
    # Failing loudly matters: a silent fallback to dense-only would make an
    # eval run report "hybrid" numbers for a retriever that never ran BM25.
    raise SystemExit(f"{name} is not importable. Install it with:  pip install langchain-classic")


def build_sparse(chunks: list[Document]):
    """BM25 over the chunk sidecar. Built once per process - at 40k chunks the
    tokenise-and-count pass takes seconds, far too slow to redo per request.
    Inverted-index implementation: see app/bm25.py for why not rank_bm25."""
    return FastBM25Retriever.from_documents(chunks)


def build_retriever(store, chunks: list[Document] | None = None, top_k: int | None = None, sparse=None):
    k = top_k or config.TOP_K
    first_k = max(config.RERANK_FETCH_K, k) if config.RERANK else k

    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": first_k,
            "fetch_k": max(config.FETCH_K, first_k * 4),
            "lambda_mult": config.MMR_LAMBDA,
        },
    )

    if config.USE_HYBRID and (chunks or sparse is not None):
        bm25 = (sparse if sparse is not None else build_sparse(chunks)).model_copy(update={"k": first_k})
        retriever = _legacy("EnsembleRetriever")(
            retrievers=[retriever, bm25],
            weights=list(config.HYBRID_WEIGHTS),
        )

    if not config.RERANK:
        return retriever

    reranker = _legacy("CrossEncoderReranker")(model=get_cross_encoder(config.RERANK_MODEL), top_n=k)
    return _legacy("ContextualCompressionRetriever")(base_compressor=reranker, base_retriever=retriever)


def dense_only_retriever(store, top_k: int | None = None):
    """Plain similarity search - the retrieval baseline for the eval harness."""
    return store.as_retriever(search_type="similarity", search_kwargs={"k": top_k or config.TOP_K})


def format_context(docs: list[Document]) -> str:
    """Number every passage so the model can cite [1], [2], ... precisely."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        location = f"{meta.get('source', 'unknown')}"
        if meta.get("page"):
            location += f", page {meta['page']}"
        blocks.append(f"[{i}] ({location})\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def citations(docs: list[Document]) -> list[dict]:
    out = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        text = doc.page_content
        out.append(
            {
                "n": i,
                "source": meta.get("source", "unknown"),
                "title": meta.get("title"),
                "section": meta.get("section"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "snippet": (text[:300] + "...") if len(text) > 300 else text,
            }
        )
    return out
