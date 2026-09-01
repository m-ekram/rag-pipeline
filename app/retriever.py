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


def build_retriever(
    store,
    chunks: list[Document] | None = None,
    top_k: int | None = None,
    *,
    use_hybrid: bool | None = None,
    mmr_lambda: float | None = None,
    weights: tuple[float, float] | None = None,
    fetch_k: int | None = None,
):
    """Build the retrieval stack. Every knob defaults to its configured value.

    The keyword overrides exist for the eval harness, which needs to vary one
    component at a time. Passing them explicitly is what keeps a variant from
    inheriting an unrelated ambient default - the bug that made the published
    ablation table unreproducible.
    """
    k = top_k or config.TOP_K
    use_hybrid = config.USE_HYBRID if use_hybrid is None else use_hybrid
    mmr_lambda = config.MMR_LAMBDA if mmr_lambda is None else mmr_lambda
    weights = config.HYBRID_WEIGHTS if weights is None else weights
    fetch_k = config.FETCH_K if fetch_k is None else fetch_k

    dense = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(fetch_k, k * 4),
            "lambda_mult": mmr_lambda,
        },
    )

    if not use_hybrid or not chunks:
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
        weights=list(weights),
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
