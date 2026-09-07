"""Document loaders for every corpus this project ingests.

All loaders are generators — the FiQA corpus alone is 57,638 documents / 46MB,
and the contamination sweep loads it repeatedly, so nothing here materialises
the whole corpus in memory unless the caller asks for it.
"""

import io
import os
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from .clean import clean_text
from .documents import Document

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# --- BEIR / FiQA --------------------------------------------------------


def load_beir_corpus(path: Optional[str] = None, *, clean: bool = True) -> Iterator[Document]:
    """Load a BEIR-format corpus.jsonl (`_id`, `title`, `text`, `metadata`).

    FiQA's titles are uniformly empty, so `title` will be "" for that corpus and
    citations fall back to the document id.
    """
    path = path or os.path.join(DATA_DIR, "fiqa", "corpus.jsonl")
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%d", path, line_no)
                continue
            text = record.get("text", "")
            yield Document(
                doc_id=str(record["_id"]),
                text=clean_text(text) if clean else text,
                title=record.get("title", "") or "",
                source="fiqa",
                metadata=record.get("metadata") or {},
            )


def load_beir_queries(path: Optional[str] = None) -> dict[str, str]:
    """Load queries.jsonl into {query_id: text}."""
    path = path or os.path.join(DATA_DIR, "fiqa", "queries.jsonl")
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            queries[str(record["_id"])] = record.get("text", "")
    return queries


def load_qrels(path: Optional[str] = None) -> dict[str, dict[str, int]]:
    """Load a BEIR qrels TSV into {query_id: {doc_id: relevance}}.

    The first line is a `query-id corpus-id score` header and is skipped.
    FiQA's test qrels are binary (every score is 1).
    """
    path = path or os.path.join(DATA_DIR, "fiqa", "qrels", "test.tsv")
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as handle:
        first = handle.readline().strip().split("\t")
        # Tolerate a missing header rather than silently dropping a judgement.
        if first and first[0] != "query-id":
            handle.seek(0)
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, score = parts[0], parts[1], parts[2]
            qrels.setdefault(query_id, {})[doc_id] = int(score)
    return qrels


# --- Noisy corpus -------------------------------------------------------


def load_noisy_corpus(path: Optional[str] = None, *, clean: bool = True) -> Iterator[Document]:
    """Load the out-of-domain distractor corpus written by eval/download_noisy_corpus.py."""
    path = path or os.path.join(DATA_DIR, "noisy_corpus", "corpus.json")
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        text = record.get("text", "")
        yield Document(
            doc_id=f"noise-{record['id']}",
            text=clean_text(text) if clean else text,
            title=record.get("title", "") or "",
            source="noisy_corpus",
        )


# --- PDF / HTML ---------------------------------------------------------


def load_pdf(
    path: str,
    *,
    clean: bool = True,
    per_page: bool = True,
    ocr_lang: str = "en",
) -> Iterator[Document]:
    """Load a PDF using dual native-text + OCR extraction.

    Native PDF text and OCR are both extracted. The preprocessing layer
    evaluates their quality and selects the most useful representation.

    Per-page documents preserve page-level provenance for citations.
    """

    from .pdf_extractor import PDFExtractor

    extractor = PDFExtractor(ocr_lang=ocr_lang)

    documents = extractor.extract(
        path,
        clean=clean,
    )

    if per_page:
        yield from documents
        return

    # If a single document is requested, combine the page-level results.
    pages = list(documents)

    if not pages:
        return

    combined_text = "\n\n".join(
        document.text
        for document in pages
        if document.text.strip()
    )

    if not combined_text.strip():
        return

    first = pages[0]

    # Preserve useful metadata while representing the document as a whole.
    metadata = {
        "extraction_method": "dual",
        "pages": len(pages),
    }

    yield Document(
        doc_id=Path(path).stem,
        text=combined_text,
        title=first.title,
        source=path,
        metadata=metadata,
    )


def load_html(path_or_html: str, *, clean: bool = True, is_markup: bool = False) -> Iterator[Document]:
    """Load an HTML file (or raw markup), split into Documents by `<h1>`/`<h2>` section.

    Sectioning here is what populates `Section Y` in citations.
    """
    from bs4 import BeautifulSoup

    if is_markup:
        markup, source, doc_id = path_or_html, "<memory>", "html"
    else:
        with open(path_or_html, encoding="utf-8", errors="replace") as handle:
            markup = handle.read()
        source = path_or_html
        doc_id = os.path.splitext(os.path.basename(path_or_html))[0]

    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else doc_id

    headings = soup.find_all(["h1", "h2"])
    if not headings:
        text = soup.get_text("\n")
        if text.strip():
            yield Document(
                doc_id=doc_id,
                text=clean_text(text) if clean else text,
                title=title,
                source=source,
            )
        return

    for index, heading in enumerate(headings):
        section = heading.get_text(strip=True)
        parts: list[str] = []
        for sibling in heading.next_siblings:
            if getattr(sibling, "name", None) in ("h1", "h2"):
                break
            chunk = sibling.get_text("\n") if hasattr(sibling, "get_text") else str(sibling)
            if chunk.strip():
                parts.append(chunk)
        text = "\n".join(parts)
        if not text.strip():
            continue
        yield Document(
            doc_id=f"{doc_id}#s{index}",
            text=clean_text(text) if clean else text,
            title=title,
            source=source,
            section=section,
        )
