"""Chunking strategies.

Two are implemented on purpose:

  naive_split      fixed-width character windows, no structural awareness.
                   This is the baseline you measure against in eval/evaluate.py.
  structured_split recursive splitting on real document boundaries plus a
                   contextual header on every chunk. This is what ships.

The contextual header is prepended to each chunk's text so the embedding, and
BM25, both see where a fragment sits. In "path" mode it is the section
breadcrumb - "[tutorial request files > Request Files > File Parameters with
UploadFile]" - rather than only the file title: a chunk that says "the timeout
is 30s" is ambiguous on its own and unambiguous with its heading attached.

PDFs arrive one Document per page. structured_split re-joins a PDF's pages
before splitting so an answer that runs across a page break stays in one
chunk, and maps each chunk back to the page it starts on for citation.
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_right

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

HEADER_MODES = {"path", "title", "none"}

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# A page break is joined as a *line* break, not a paragraph break: the splitter
# treats "\n\n" as a boundary it splits on before anything finer, which would
# turn every page edge back into a hard chunk edge.
_PAGE_JOIN = "\n"


def _chunk_id(text: str, source: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text[:200]}".encode()).hexdigest()
    return digest[:16]


def _header(meta: dict, mode: str) -> str:
    title = meta.get("title") or meta.get("source", "")
    label = title
    if mode == "path" and meta.get("section"):
        label = f"{title} > {meta['section']}"
    page = meta.get("page")
    return f"[{label} - p.{page}]" if page else f"[{label}]"


def _finalise(chunks: list[Document], header_mode: str) -> list[Document]:
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

        content = f"{_header(meta, header_mode)}\n{text}" if header_mode != "none" else text
        out.append(Document(page_content=content, metadata=meta))
    return out


def _heading_index(text: str) -> tuple[list[int], list[str]]:
    """Offsets of markdown headings and the breadcrumb in force after each.

    Headings inside fenced code are skipped: a Python comment is not a section.
    """
    offsets: list[int] = []
    paths: list[str] = []
    stack: list[str] = []
    in_code = False
    pos = 0
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            in_code = not in_code
        elif not in_code:
            match = _HEADING.match(line.rstrip("\n"))
            if match:
                level = len(match.group(1))
                stack = stack[: level - 1] + [match.group(2).strip()]
                offsets.append(pos)
                paths.append(" > ".join(stack))
        pos += len(line)
    return offsets, paths


def _merge_pdf_pages(docs: list[Document]) -> list[Document]:
    """Join consecutive pages of one PDF into a single Document.

    `_page_starts` records (char offset, page, section) so chunks can be mapped
    back to the page they start on.
    """
    out: list[Document] = []
    group: list[Document] = []

    def flush() -> None:
        if not group:
            return
        starts, pos = [], 0
        for doc in group:
            starts.append((pos, doc.metadata["page"], doc.metadata.get("section")))
            pos += len(doc.page_content) + len(_PAGE_JOIN)
        meta = {k: v for k, v in group[0].metadata.items() if k not in {"page", "section"}}
        meta["_page_starts"] = starts
        out.append(Document(page_content=_PAGE_JOIN.join(d.page_content for d in group), metadata=meta))
        group.clear()

    for doc in docs:
        if doc.metadata.get("page") is None:
            flush()
            out.append(doc)
            continue
        if group and group[-1].metadata.get("source") != doc.metadata.get("source"):
            flush()
        group.append(doc)
    flush()
    return out


def structured_split(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    add_headers: bool = True,
    header_mode: str | None = None,
) -> list[Document]:
    mode = (header_mode or config.HEADER_MODE) if add_headers else "none"
    if mode not in HEADER_MODES:
        raise ValueError(f"header_mode must be one of {sorted(HEADER_MODES)}, got {mode!r}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        keep_separator=True,
        add_start_index=True,
    )

    pieces: list[Document] = []
    for doc in _merge_pdf_pages(docs):
        meta = dict(doc.metadata)
        page_starts = meta.pop("_page_starts", None)
        if page_starts:
            page_offsets = [p[0] for p in page_starts]
        else:
            head_offsets, head_paths = _heading_index(doc.page_content)

        for piece in splitter.split_documents([Document(page_content=doc.page_content, metadata=meta)]):
            start = max(0, piece.metadata.pop("start_index", 0))
            if page_starts:
                first = page_starts[max(0, bisect_right(page_offsets, start) - 1)]
                last = page_starts[max(0, bisect_right(page_offsets, start + len(piece.page_content) - 1) - 1)]
                piece.metadata["page"] = first[1]
                if last[1] != first[1]:
                    piece.metadata["page_end"] = last[1]
                if first[2]:
                    piece.metadata["section"] = first[2]
            else:
                i = bisect_right(head_offsets, start) - 1
                if i >= 0:
                    piece.metadata["section"] = head_paths[i]
            pieces.append(piece)

    return _finalise(pieces, mode)


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
    return _finalise(splitter.split_documents(docs), "none")


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
