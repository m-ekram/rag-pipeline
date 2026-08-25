"""Retrieval.

Dense-only search misses exact identifiers - error codes, CLI flags, config
keys - because embeddings smooth them away. BM25 nails those and misses
paraphrases. Running both and fusing the ranks is what moves top-k relevance on
technical documentation, so hybrid is the default.

MMR sits on top of the dense leg: it pulls FETCH_K candidates and then picks
TOP_K that are relevant *and* mutually dissimilar, which stops all five slots
being filled by near-duplicate paragraphs of the same page.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def build_retriever(store, chunks: list[Document] | None = None, top_k: int | None = None):
    k = top_k or config.TOP_K

    dense = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(config.FETCH_K, k * 4),
            "lambda_mult": config.MMR_LAMBDA,
        },
    )

    if not config.USE_HYBRID or not chunks:
        return dense

    try:
        from langchain_community.retrievers import BM25Retriever

        try:  # langchain >= 1.0 moved the legacy retrievers into langchain-classic
            from langchain_classic.retrievers import EnsembleRetriever
        except ImportError:
            from langchain.retrievers import EnsembleRetriever
    except ImportError:
        logger.warning("rank-bm25 / EnsembleRetriever unavailable - falling back to dense-only retrieval")
        return dense

    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = k

    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=list(config.HYBRID_WEIGHTS),
    )


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
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "snippet": (text[:300] + "...") if len(text) > 300 else text,
            }
        )
    return out
