"""Turn a directory of mixed documents into LangChain Documents.

Each returned Document carries enough metadata to build a real citation:
  source    relative path, e.g. "manuals/networking.pdf"
  page      1-based page number (PDFs only)
  title     human-readable document title
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".docx"}

# Collapse the ragged whitespace PDF extraction leaves behind; it wrecks both
# chunk boundaries and BM25 tokenisation if left in.
_WS = re.compile(r"[ \t\u00a0]+")
_BLANKS = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words split across lines
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def _title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def _load_pdf(path: Path, source: str) -> list[Document]:
    from pypdf import PdfReader

    docs: list[Document] = []
    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata else None) or _title_from_path(path)
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception as exc:  # a single corrupt page shouldn't kill the run
            logger.warning("%s page %d: extraction failed (%s)", source, i, exc)
            continue
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": source, "page": i, "title": str(title)},
                )
            )
    return docs


def _load_html(path: Path, source: str) -> list[Document]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    title = (soup.title.string if soup.title and soup.title.string else None) or _title_from_path(path)
    text = clean_text(soup.get_text("\n"))
    return [Document(page_content=text, metadata={"source": source, "title": str(title)})] if text else []


def _load_docx(path: Path, source: str) -> list[Document]:
    import docx  # python-docx

    document = docx.Document(str(path))
    text = clean_text("\n".join(p.text for p in document.paragraphs))
    return [Document(page_content=text, metadata={"source": source, "title": _title_from_path(path)})] if text else []


def _load_plain(path: Path, source: str) -> list[Document]:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [Document(page_content=text, metadata={"source": source, "title": _title_from_path(path)})] if text else []


def load_file(path: Path, root: Path) -> list[Document]:
    source = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, source)
    if suffix in {".html", ".htm"}:
        return _load_html(path, source)
    if suffix == ".docx":
        return _load_docx(path, source)
    return _load_plain(path, source)


def load_directory(data_dir: str | Path) -> list[Document]:
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Data directory not found: {root}")

    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise SystemExit(
            f"No supported documents in {root}. "
            f"Drop some {', '.join(sorted(SUPPORTED_SUFFIXES))} files in there and re-run."
        )

    docs: list[Document] = []
    for path in paths:
        try:
            loaded = load_file(path, root)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        logger.info("Loaded %-45s %3d section(s)", path.relative_to(root).as_posix(), len(loaded))
        docs.extend(loaded)

    logger.info("Loaded %d section(s) from %d file(s)", len(docs), len(paths))
    return docs
