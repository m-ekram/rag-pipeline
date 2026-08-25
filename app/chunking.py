"""Chunking strategies.

Two are implemented on purpose:

  naive_split      fixed-width character windows, no structural awareness.
                   This is the baseline you measure against in eval/evaluate.py.
  structured_split recursive splitting on real document boundaries plus a
                   contextual header on every chunk. This is what ships.

The contextual header ("Networking Guide - p.12") is prepended to each chunk's
text so the embedding, and BM25, both see which document a fragment came from.
It is the cheapest retrieval win available: a chunk that says "the timeout is
30s" is ambiguous on its own and unambiguous with two words of context.
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

import config

# Ordered coarse -> fine. The splitter walks this list and only falls through to
# a finer separator when a piece is still oversized, so paragraphs stay intact
# whenever they fit.
SEPARATORS = [
    "\n## ",   # markdown section
    "\n### ",
    "\n\n",    # paragraph
    "\n",      # line
    ". ",      # sentence
    " ",
    "",
]


def _chunk_id(text: str, source: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text[:200]}".encode()).hexdigest()
    return digest[:16]


def _header(meta: dict) -> str:
    title = meta.get("title") or meta.get("source", "")
    page = meta.get("page")
    return f"[{title} - p.{page}]" if page else f"[{title}]"


def _finalise(chunks: list[Document], add_headers: bool) -> list[Document]:
    out: list[Document] = []
    per_source: dict[str, int] = {}
    for chunk in chunks:
        text = chunk.page_content.strip()
        if len(text) < config.MIN_CHUNK_CHARS:
            continue
        source = chunk.metadata.get("source", "unknown")
        index = per_source.get(source, 0)
        per_source[source] = index + 1

        meta = dict(chunk.metadata)
        meta["chunk_index"] = index
        meta["chunk_id"] = _chunk_id(text, source, index)
        meta["char_count"] = len(text)

        content = f"{_header(meta)}\n{text}" if add_headers else text
        out.append(Document(page_content=content, metadata=meta))
    return out


def structured_split(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    add_headers: bool = True,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        keep_separator=True,
    )
    return _finalise(splitter.split_documents(docs), add_headers)


def naive_split(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
) -> list[Document]:
    """Fixed-width windows with no overlap and no headers - the baseline."""
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return _finalise(splitter.split_documents(docs), add_headers=False)


def stats(chunks: list[Document]) -> dict:
    if not chunks:
        return {"count": 0}
    sizes = sorted(c.metadata["char_count"] for c in chunks)
    return {
        "count": len(chunks),
        "sources": len({c.metadata.get("source") for c in chunks}),
        "min_chars": sizes[0],
        "median_chars": sizes[len(sizes) // 2],
        "max_chars": sizes[-1],
        "mean_chars": round(sum(sizes) / len(sizes)),
    }
