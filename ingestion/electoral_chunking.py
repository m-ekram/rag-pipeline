"""Structure-aware chunking for Indian electoral rolls.

Why fixed-size windows do badly here, from the OCR output:

    विधानसभा निर्वाचन क्षेत्र की संख्या एवं नाम: 183-कुम्हरार   <- page header
    भाग संख्या:: 1
    अनुभाग संख्या एवं नाम: 1-गोविन्द मित्रा रोड
    151]  SHS5213004  152]  SHS0796649  153]  SHS0582973      <- one row-band
    निर्वाचक का नाम: ...  निर्वाचक का नाम: ...  निर्वाचक का नाम: ...

The page is a three-column table, and OCR flattens it column-major: three serial
numbers, then three EPIC numbers, then three names. Fields are interleaved
across records, so a chunk boundary drawn every 180 words lands mid-band and
splits voters from their own EPIC numbers.

Two changes, both measurable:

1. **Split on row-band boundaries** (a serial/EPIC marker) instead of a word
   count, so a chunk holds whole bands.
2. **Repeat the page header on every chunk.** A bare list of names carries no
   constituency, part or section, which is why so many chunks look alike to an
   embedder and why citations from mid-page chunks were unusable.

Per-voter chunking is deliberately *not* attempted: at this OCR quality fields
are dropped often enough that positional re-association would silently attach
the wrong father's name to a voter. Band-level is the honest granularity.
"""

import re
from typing import Iterator, Optional

from .documents import Chunk, Document

# A record marker: "151]" or "154" followed by an EPIC id like SHS5213004 or
# BR/35/207/267110. OCR frequently drops the bracket, so it is optional.
_RECORD = re.compile(r"^\s*(\d{1,5})\s*[\]\)]?\s*$")
_EPIC = re.compile(r"^\s*([A-Z]{2,4}\d{6,10}|[A-Z]{2}/\d{2}/\d{3}/\d{4,8})\s*$")

# Header lines that identify where a chunk sits in the roll.
_HEADER_PATTERNS = (
    re.compile(r"विधानसभा[^\n]*?\d{2,3}\s*-\s*\S+"),   # constituency no. + name
    re.compile(r"भाग\s*संख्या[^\n]*"),                    # part number
    re.compile(r"अनुभाग\s*संख्या[^\n]*"),                 # section number + name
)
_WORD = re.compile(r"\S+")


def extract_header(text: str, *, max_lines: int = 12) -> str:
    """Pull the constituency / part / section lines from the top of a page."""
    found: list[str] = []
    for line in text.splitlines()[:max_lines]:
        line = line.strip()
        if not line:
            continue
        for pattern in _HEADER_PATTERNS:
            if pattern.search(line):
                if line not in found:
                    found.append(line)
                break
    return " | ".join(found)


def extract_epics(text: str) -> list[str]:
    """Every EPIC (voter ID) in a chunk — useful as searchable metadata."""
    return _EPIC.findall(text) or [
        m.group(1) for line in text.splitlines() if (m := _EPIC.match(line))
    ]


def _is_band_start(lines: list[str], index: int) -> bool:
    """A serial number immediately followed by an EPIC starts a new row-band."""
    if not _RECORD.match(lines[index]):
        return False
    for offset in (1, 2):
        if index + offset < len(lines) and _EPIC.match(lines[index + offset]):
            return True
    return False


class ElectoralRecordChunker:
    """Chunk an electoral-roll page on row-band boundaries, header attached."""

    def __init__(
        self,
        bands_per_chunk: int = 2,
        *,
        include_header: bool = True,
        max_words: int = 320,
    ):
        if bands_per_chunk <= 0:
            raise ValueError("bands_per_chunk must be positive")
        self.bands_per_chunk = bands_per_chunk
        self.include_header = include_header
        self.max_words = max_words
        self.name = f"electoral-{bands_per_chunk}band"

    def _bands(self, lines: list[str]) -> list[list[str]]:
        starts = [i for i in range(len(lines)) if _is_band_start(lines, i)]
        if not starts:
            return [lines] if lines else []

        bands: list[list[str]] = []
        preamble = lines[: starts[0]]
        if preamble:
            bands.append(preamble)
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(lines)
            bands.append(lines[start:end])
        return bands

    def chunk(self, document: Document) -> Iterator[Chunk]:
        lines = [ln for ln in document.text.splitlines() if ln.strip()]
        if not lines:
            return

        header = extract_header(document.text) if self.include_header else ""
        bands = self._bands(lines)

        ordinal = 0
        consumed = 0
        for start in range(0, len(bands), self.bands_per_chunk):
            group = [ln for band in bands[start:start + self.bands_per_chunk]
                     for ln in band]
            if not group:
                continue

            body = "\n".join(group)
            # The header is context, not content: it is prepended so the chunk is
            # self-describing, but it is not counted against the word budget.
            text = f"{header}\n{body}" if header and not body.startswith(header) else body

            words = len(_WORD.findall(body))
            epics = extract_epics(body)

            yield Chunk(
                chunk_id=f"{document.doc_id}::{ordinal}",
                doc_id=document.doc_id,
                text=text,
                ordinal=ordinal,
                title=document.title,
                source=document.source,
                page=document.page,
                section=header or document.section,
                start_word=consumed,
                end_word=consumed + words,
                metadata={**document.metadata, "epics": epics, "n_epics": len(epics)},
            )
            ordinal += 1
            consumed += words
