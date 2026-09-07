"""Chunking strategies.

Phase 1 Core requires two: fixed-size and sentence-aware. Both are word-based
rather than character-based so `chunk_size` maps predictably onto the embedding
model's token limit (bge-small tops out at 512 tokens, roughly 350-400 words).

Both strategies preserve the parent document's citation metadata and record the
word span they came from, so a chunk can always be located in its source.
"""

import re
from typing import Iterator, Optional, Protocol

from .documents import Chunk, Document

DEVA_TO_ARABIC = str.maketrans("०१२३४५६७८९", "0123456789")
ARABIC_TO_DEVA = str.maketrans("0123456789", "०१२३४५६७८९")


def canonicalize_house_number(raw_house: str) -> str:
    """
    Cleans up and normalizes a house number string:
    - Strips 'फोटो उपलब्ध', 'फोटो', pipes, extra whitespace
    - Normalizes Devanagari numerals to Arabic digits
    """
    if not raw_house:
        return ""

    # 1. Strip common OCR artifacts found next to house number box
    cleaned = re.sub(r"(?:फोटो\s*उपलब्ध|फोटो|उपलब्ध|[|:;._])", " ", raw_house)
    cleaned = cleaned.strip()

    # If empty after stripping, return empty
    if not cleaned:
        return ""

    # 2. Extract just the core house number tokens (e.g. '3', '28/A', '१२-ख')
    tokens = cleaned.split()
    core_house = tokens[0] if tokens else cleaned

    # 3. Translate Devanagari digits (०-९ -> 0-9)
    return core_house.translate(DEVA_TO_ARABIC)


def dual_house_number_display(house_str: str) -> str:
    """Return dual Arabic and Devanagari representations for 100% search recall."""
    if not house_str:
        return ""
    clean = canonicalize_house_number(house_str)
    if not clean:
        return ""
    arabic = clean.translate(DEVA_TO_ARABIC)
    deva = clean.translate(ARABIC_TO_DEVA)
    if arabic != deva:
        return f"{arabic} / {deva}"
    return clean


# Sentence splitting without an NLTK dependency. Python's `re` only supports
# fixed-width lookbehind, so abbreviations cannot be excluded inline — instead
# their periods are swapped for a sentinel, the split runs, and they are
# restored afterwards.
_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St",
    "Inc", "Ltd", "Co", "Corp", "vs", "etc", "e.g", "i.e", "approx",
    "Fig", "Vol", "pp", "Jan", "Feb", "Mar", "Apr", "Jun",
    "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
)
_SENTINEL = "\x00"
_ABBR_RE = re.compile(rf"\b(?:{'|'.join(re.escape(a) for a in _ABBREVIATIONS)})\.")
# A lone capital followed by a period is an initial ("J. Smith"), not an ending.
_INITIAL_RE = re.compile(r"\b[A-Z]\.")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")


def _protect(text: str) -> str:
    text = _ABBR_RE.sub(lambda m: m.group(0)[:-1] + _SENTINEL, text)
    return _INITIAL_RE.sub(lambda m: m.group(0)[:-1] + _SENTINEL, text)


def _restore(text: str) -> str:
    return text.replace(_SENTINEL, ".")


_WORD = re.compile(r"\S+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without an NLTK dependency."""
    if not text.strip():
        return []
    # Paragraph breaks are hard boundaries regardless of punctuation.
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        protected = _protect(paragraph)
        for part in _SENTENCE_BOUNDARY.split(protected):
            part = _restore(part).strip()
            if part:
                sentences.append(part)
    return sentences


class Chunker(Protocol):
    """Interface both strategies satisfy, so experiments can swap them by config."""

    name: str

    def chunk(self, document: Document) -> Iterator[Chunk]: ...


def _build(document: Document, ordinal: int, words: list[str],
           start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id}::{ordinal}",
        doc_id=document.doc_id,
        text=" ".join(words),
        ordinal=ordinal,
        title=document.title,
        source=document.source,
        page=document.page,
        section=document.section,
        start_word=start,
        end_word=end,
        metadata=dict(document.metadata),
    )


class FixedSizeChunker:
    """Fixed word-count windows, optionally overlapping.

    `overlap=0` is the Phase 1 Core strategy; a non-zero overlap is the Stretch
    "overlapping fixed-size" variant and needs no other code change.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.name = f"fixed-{chunk_size}-o{overlap}"

    def chunk(self, document: Document) -> Iterator[Chunk]:
        words = _WORD.findall(document.text)
        if not words:
            return

        step = self.chunk_size - self.overlap
        ordinal = 0
        for start in range(0, len(words), step):
            window = words[start:start + self.chunk_size]
            if not window:
                break
            yield _build(document, ordinal, window, start, start + len(window))
            ordinal += 1
            # A final short window is emitted above; stop before re-emitting it.
            if start + self.chunk_size >= len(words):
                break


class SentenceAwareChunker:
    """Pack whole sentences up to `max_words`, never splitting mid-sentence.

    A sentence longer than `max_words` is emitted alone rather than truncated.
    `hard_max_words` is the safety valve: real corpora contain pathological
    "sentences" (unpunctuated walls of text — FiQA has one of 1,264 words), and
    a chunk past the embedding model's token limit is silently truncated at
    encode time, so most of it becomes invisible to dense retrieval. Force-
    splitting those keeps the fixed-vs-sentence comparison a comparison of
    strategies rather than of truncation artefacts.
    """

    def __init__(self, max_words: int = 200, hard_max_words: int = 350):
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        if hard_max_words < max_words:
            raise ValueError("hard_max_words must be >= max_words")
        self.max_words = max_words
        self.hard_max_words = hard_max_words
        self.name = f"sentence-{max_words}"

    def chunk(self, document: Document) -> Iterator[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return

        ordinal = 0
        buffer: list[str] = []
        buffer_words = 0
        start_word = 0
        consumed = 0

        for sentence in sentences:
            words = _WORD.findall(sentence)
            n = len(words)

            if n > self.hard_max_words:
                # Flush what is buffered, then window the oversized sentence.
                if buffer:
                    yield _build(document, ordinal, _WORD.findall(" ".join(buffer)),
                                 start_word, consumed)
                    ordinal += 1
                    buffer, buffer_words, start_word = [], 0, consumed
                for offset in range(0, n, self.hard_max_words):
                    window = words[offset:offset + self.hard_max_words]
                    yield _build(document, ordinal, window,
                                 consumed + offset, consumed + offset + len(window))
                    ordinal += 1
                consumed += n
                start_word = consumed
                continue

            if buffer and buffer_words + n > self.max_words:
                yield _build(document, ordinal, _WORD.findall(" ".join(buffer)),
                             start_word, consumed)
                ordinal += 1
                buffer, buffer_words, start_word = [], 0, consumed
            buffer.append(sentence)
            buffer_words += n
            consumed += n

        if buffer:
            yield _build(document, ordinal, _WORD.findall(" ".join(buffer)),
                         start_word, consumed)


def linearize_electoral_summary_table(text: str) -> str:
    """
    Parses the messy OCR table grid at the bottom of Page 1:
    'आरंभिक क्रम अंतिम क्रम पुरुष महिला तृतीय लिंग कुल'
    '1 1031 528 494 0 1022'
    and linearizes it into unambiguous structured record text so LLMs
    never transpose table columns or confuse serials with totals.
    """
    import re

    # Check if text contains the summary table keywords
    if not ("मतदाताओं की संख्या" in text or "आरंभिक क्रम" in text or "पुरुष" in text):
        return text

    # Look for the row containing the 5 or 6 digits at the end of the summary table
    # Matches patterns like: [।] 703/1031 528 494 0 1022/2022
    pattern = re.compile(
        r'([०-९0-9।|]+)\s+([०-९0-9]+)\s+([०-९0-9]+)\s+([०-९0-9]+)\s+([०-९0-9]+)\s+([०-९0-9]+)'
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return text

    # In electoral summary tables, if multiple rows exist (e.g. sections + total),
    # the last row represents the grand total for the polling station.
    match = matches[-1]
    start_serial, end_serial, male, female, third_gender, total = match.groups()

    # Devanagari to Arabic translation table (0-9 plus Devanagari danda / pipe to 1)
    trans = str.maketrans("०१२३४५६७८९।|", "012345678911")
    start_clean = start_serial.translate(trans).strip()
    end_clean = end_serial.translate(trans).strip()
    male_clean = male.translate(trans).strip()
    female_clean = female.translate(trans).strip()
    third_clean = third_gender.translate(trans).strip()
    total_clean = total.translate(trans).strip()

    # Self-consistency check: Total is usually Male + Female
    try:
        m_val = int(male_clean)
        f_val = int(female_clean)
        t_val = int(total_clean)
        if abs((m_val + f_val) - t_val) > 10 and abs((m_val + f_val) - (t_val - 1000)) <= 2:
            # Fixes OCR reading '1022' as '2022' due to merged border line
            total_clean = str(m_val + f_val)
    except Exception:
        pass

    linearized_block = f"""
### [सारणीबद्ध सारांश रिकॉर्ड / Labeled Summary Record]
- आरंभिक क्रम संख्या: {start_clean}
- अंतिम क्रम संख्या: {end_clean}
- पुरुष मतदाताओं की कुल संख्या: {male_clean}
- महिला मतदाताओं की कुल संख्या: {female_clean}
- तृतीय लिंग मतदाताओं की संख्या: {third_clean}
- मतदाताओं की कुल संख्या (Total Voters): {total_clean}
"""
    # Append the explicit linearized record to the chunk text
    return text + "\n" + linearized_block.strip()


class ElectoralRecordChunker:
    """Specialized chunker for tabular electoral roll pages.

    Detects multi-column voter card layouts and binds each voter's details
    (Name, Relation, House, Age, Gender, EPIC) into atomic key-value records.
    Packs a fixed number of complete voter records per chunk so no voter's
    attributes are severed across chunk boundaries.

    Falls back to `SentenceAwareChunker` or `FixedSizeChunker` for non-electoral pages.
    """

    def __init__(
        self,
        records_per_chunk: int = 5,
        fallback_chunker: Optional[Chunker] = None,
        group_by_household: bool = True,
    ):
        self.records_per_chunk = records_per_chunk
        self.fallback_chunker = fallback_chunker or SentenceAwareChunker(max_words=200)
        self.group_by_household = group_by_household
        self.name = "electoral-household" if group_by_household else f"electoral-{records_per_chunk}"

    def chunk(self, document: Document) -> Iterator[Chunk]:
        from .electoral import is_electoral_text, parse_electoral_records

        if not is_electoral_text(document.text):
            yield from self.fallback_chunker.chunk(document)
            return

        # Page 1 contains polling station metadata, constituency details, and elector summaries
        page_num = getattr(document, "page", None) or document.metadata.get("page")
        if page_num == 1:
            chunk_text = linearize_electoral_summary_table(document.text)
            full_text = f"### निर्वाचन नामावली एवं मतदान केंद्र विवरण (Polling Station Metadata)\n{chunk_text}"
            words = _WORD.findall(full_text)
            chunk_id = (
                f"{document.doc_id}::metadata"
                if "#p1" in document.doc_id
                else f"{document.doc_id}#p1::metadata"
            )
            yield Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=full_text,
                ordinal=0,
                title=document.title,
                source=document.source,
                page=1,
                section="Polling Station Metadata",
                start_word=0,
                end_word=len(words),
                metadata={
                    **dict(document.metadata),
                    "block_type": "electoral_metadata",
                    "chunk_type": "electoral_metadata",
                    "page_num": 1,
                    "doc_id": document.doc_id,
                },
            )
            return

        header_ctx, records = parse_electoral_records(document.text)
        if not records:
            # Check if this page has an electoral summary table (e.g. final summary page)
            summary_linearized = linearize_electoral_summary_table(document.text)
            if summary_linearized != document.text:
                full_text = f"### निर्वाचन नामावली सारांश (Electoral Roll Summary)\n{summary_linearized}"
                words = _WORD.findall(full_text)
                yield Chunk(
                    chunk_id=f"{document.doc_id}::summary",
                    doc_id=document.doc_id,
                    text=full_text,
                    ordinal=0,
                    title=document.title,
                    source=document.source,
                    page=page_num or document.metadata.get("page", 0),
                    section="Electoral Roll Summary",
                    start_word=0,
                    end_word=len(words),
                    metadata={
                        **dict(document.metadata),
                        "block_type": "electoral_summary",
                        "chunk_type": "electoral_summary",
                        "page_num": page_num or document.metadata.get("page", 0),
                        "doc_id": document.doc_id,
                    },
                )
                return
            yield from self.fallback_chunker.chunk(document)
            return

        header_prefix = f"[{header_ctx}]\n" if header_ctx else ""

        if self.group_by_household:
            from collections import defaultdict

            # 1. Group cards on this page by House Number (मकान संख्या)
            households: dict[str, list] = defaultdict(list)
            for rec in records:
                raw_h = (rec.house or "").strip()
                h_val = canonicalize_house_number(raw_h)
                house_key = h_val if h_val else "UNKNOWN"
                households[house_key].append(rec)

            ordinal = 0
            seen_ids: set[str] = set()

            for house_num, member_records in households.items():
                # Build the unified Household Parent Context with dual Devanagari/Arabic normalizer
                dual_house = dual_house_number_display(house_num)
                dual_suffix = f" [मकान संख्या: {dual_house}]" if dual_house and dual_house != house_num else ""
                if house_num != "UNKNOWN":
                    household_parent_id = f"{document.doc_id}#house_{house_num}"
                    parent_household_text = (
                        f"{header_prefix}### परिवार / मकान संख्या: {house_num} (कुल सदस्य: {len(member_records)}){dual_suffix}\n"
                        + "\n".join(r.to_markdown() for r in member_records)
                    )
                else:
                    household_parent_id = f"{document.doc_id}#house_unknown"
                    parent_household_text = (
                        f"{header_prefix}### अनिर्दिष्ट मकान संख्या (Unspecified House) (कुल सदस्य: {len(member_records)})\n"
                        + "\n".join(r.to_markdown() for r in member_records)
                    )

                # 2. Emit each card as a Child Chunk linked to the Household Parent
                for rec in member_records:
                    dual_note = f" (House: {dual_house})" if dual_house else ""
                    child_text = f"{header_prefix}{rec.to_markdown()}{dual_note}"
                    words = _WORD.findall(child_text)

                    base_cid = f"{document.doc_id}::s{rec.serial}" if rec.serial else f"{document.doc_id}::{ordinal}"
                    cid = base_cid
                    if cid in seen_ids:
                        cid = f"{base_cid}_{ordinal}"
                    seen_ids.add(cid)

                    yield Chunk(
                        chunk_id=cid,
                        doc_id=document.doc_id,
                        text=child_text,
                        ordinal=ordinal,
                        title=document.title,
                        source=document.source,
                        page=document.page,
                        section=document.section or (f"House {house_num}" if house_num != "UNKNOWN" else "Voters"),
                        start_word=0,
                        end_word=len(words),
                        metadata={
                            **dict(document.metadata),
                            "block_type": "electoral_voter",
                            "chunk_type": "electoral_voter",
                            "house_num": house_num,
                            "serial": rec.serial,
                            "epic": rec.epic,
                            "voter_name": rec.name,
                            "relation": rec.relation,
                            "age": rec.age,
                            "gender": rec.gender,
                            "parent_id": household_parent_id,
                            "parent_text": parent_household_text,
                            "voter_count": len(member_records),
                        },
                    )
                    ordinal += 1
            return

        # Sequential batching fallback when group_by_household=False
        ordinal = 0
        for i in range(0, len(records), self.records_per_chunk):
            batch = records[i : i + self.records_per_chunk]
            chunk_lines = []
            if header_ctx:
                chunk_lines.append(f"[{header_ctx}]")
            for rec in batch:
                chunk_lines.append(rec.to_markdown())

            chunk_text = "\n".join(chunk_lines)
            words = _WORD.findall(chunk_text)

            yield Chunk(
                chunk_id=f"{document.doc_id}::{ordinal}",
                doc_id=document.doc_id,
                text=chunk_text,
                ordinal=ordinal,
                title=document.title,
                source=document.source,
                page=document.page,
                section=document.section,
                start_word=0,
                end_word=len(words),
                metadata={
                    **dict(document.metadata),
                    "chunk_type": "electoral_records",
                    "voter_count": len(batch),
                },
            )
            ordinal += 1



class StructureAwareParentChildChunker:
    """Structure-aware hierarchical chunker.

    Identifies tabular structures vs. text sections:
    - Tables: Preserves full table as Parent; chunks rows into Children with persistent column headers prepended.
    - Text: Paragraph children (~120 words) pointing to Parent section context (~500 words).
    - Stores `parent_id` and `parent_text` in child metadata for LLM prompt context expansion.
    """

    def __init__(
        self,
        rows_per_child: int = 3,
        text_child_words: int = 120,
        text_parent_words: int = 500,
    ):
        self.rows_per_child = rows_per_child
        self.text_child_words = text_child_words
        self.text_parent_words = text_parent_words
        self.name = "parent-child-structure"

    def chunk(self, document: Document) -> Iterator[Chunk]:
        from .electoral import is_electoral_text

        # Electoral documents have their own atomic card strategy
        if is_electoral_text(document.text):
            yield from ElectoralRecordChunker(records_per_chunk=5).chunk(document)
            return

        caption_regex = re.compile(r"^(?:Table|Tab\.?|Annexure|Schedule)\s*\d+[:.\s-]", re.IGNORECASE)
        lines = document.text.splitlines()
        i = 0
        child_ordinal = 0
        parent_ordinal = 0

        while i < len(lines):
            line = lines[i]

            # Detect if current line is a table caption (e.g. '### Table 26...' or 'Table 26...')
            table_caption = ""
            trimmed = line.strip()

            if trimmed.startswith("#") or caption_regex.search(trimmed):
                next_k = i + 1
                while next_k < len(lines) and not lines[next_k].strip():
                    next_k += 1
                if (
                    next_k + 1 < len(lines)
                    and lines[next_k].strip().startswith("|")
                    and re.match(r"^\s*\|\s*[-:]+", lines[next_k + 1])
                ):
                    table_caption = trimmed.lstrip("#").strip()
                    i = next_k
                    line = lines[i]

            # Detect Markdown table start: e.g. line starts with '|' and next line has '| ---'
            if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|\s*[-:]+", lines[i + 1]):
                table_lines = []
                if table_caption:
                    table_lines.append(f"### {table_caption}")
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1

                if len(table_lines) >= 2:
                    if table_caption:
                        header_line = table_lines[1]
                        sep_line = table_lines[2]
                        data_rows = table_lines[3:]
                    else:
                        header_line = table_lines[0]
                        sep_line = table_lines[1]
                        data_rows = table_lines[2:]

                    stitched_info = None
                    stitched_tables = document.metadata.get("stitched_tables", {})
                    if table_caption and table_caption in stitched_tables:
                        stitched_info = stitched_tables[table_caption]
                    elif not table_caption and stitched_tables:
                        stitched_info = next(iter(stitched_tables.values()))

                    if stitched_info:
                        parent_text = stitched_info["parent_text"]
                        parent_id = stitched_info["parent_id"]
                        if not table_caption:
                            table_caption = stitched_info.get("caption", "")
                    else:
                        parent_text = "\n".join(table_lines)
                        parent_id = f"{document.doc_id}::p{parent_ordinal}"
                        parent_ordinal += 1

                    caption_prefix = f"### {table_caption}\n" if table_caption else ""

                    if not data_rows:
                        # Table with only header
                        yield Chunk(
                            chunk_id=f"{document.doc_id}::{child_ordinal}",
                            doc_id=document.doc_id,
                            text=parent_text,
                            ordinal=child_ordinal,
                            title=document.title,
                            source=document.source,
                            page=document.page,
                            section=document.section,
                            metadata={
                                **dict(document.metadata),
                                "block_type": "table",
                                "caption": table_caption,
                                "parent_id": parent_id,
                                "parent_text": parent_text,
                            },
                        )
                        child_ordinal += 1
                    else:
                        from .layout import linearize_table_row
                        raw_headers = [c.strip() for c in header_line.split("|")[1:-1]]
                        for r_idx in range(0, len(data_rows), self.rows_per_child):
                            batch_rows = data_rows[r_idx : r_idx + self.rows_per_child]
                            record_lines = []
                            for row in batch_rows:
                                cells = [c.strip() for c in row.split("|")[1:-1]] if row.strip().startswith("|") else []
                                if cells and raw_headers:
                                    rec_str = linearize_table_row(raw_headers, cells)
                                    if rec_str:
                                        record_lines.append(rec_str)
                            record_prefix = "\n".join(record_lines) + "\n\n" if record_lines else ""
                            child_text = f"{caption_prefix}{record_prefix}{header_line}\n{sep_line}\n" + "\n".join(batch_rows)
                            yield Chunk(
                                chunk_id=f"{document.doc_id}::{child_ordinal}",
                                doc_id=document.doc_id,
                                text=child_text,
                                ordinal=child_ordinal,
                                title=document.title,
                                source=document.source,
                                page=document.page,
                                section=document.section,
                                metadata={
                                    **dict(document.metadata),
                                    "block_type": "table",
                                    "caption": table_caption,
                                    "parent_id": parent_id,
                                    "parent_text": parent_text,
                                },
                            )
                            child_ordinal += 1
                continue

            # Accumulate text paragraph block
            text_lines = []
            while i < len(lines):
                cur = lines[i]
                cur_trimmed = cur.strip()
                if cur_trimmed.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|\s*[-:]+", lines[i + 1]):
                    break
                if cur_trimmed.startswith("#") or caption_regex.search(cur_trimmed):
                    next_k = i + 1
                    while next_k < len(lines) and not lines[next_k].strip():
                        next_k += 1
                    if (
                        next_k + 1 < len(lines)
                        and lines[next_k].strip().startswith("|")
                        and re.match(r"^\s*\|\s*[-:]+", lines[next_k + 1])
                    ):
                        break
                text_lines.append(cur)
                i += 1

            block_text = "\n".join(text_lines).strip()
            if not block_text:
                continue

            # Create parent for this text section
            parent_text = block_text
            parent_id = f"{document.doc_id}::p{parent_ordinal}"
            parent_ordinal += 1

            sentences = split_sentences(block_text)
            current_words: list[str] = []
            current_sentences: list[str] = []

            def emit_child(s_list: list[str], c_ord: int) -> Chunk:
                c_text = " ".join(s_list)
                return Chunk(
                    chunk_id=f"{document.doc_id}::{c_ord}",
                    doc_id=document.doc_id,
                    text=c_text,
                    ordinal=c_ord,
                    title=document.title,
                    source=document.source,
                    page=document.page,
                    section=document.section,
                    metadata={
                        **dict(document.metadata),
                        "block_type": "text",
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                    },
                )

            for s in sentences:
                s_words = _WORD.findall(s)
                if len(current_words) + len(s_words) > self.text_child_words and current_sentences:
                    yield emit_child(current_sentences, child_ordinal)
                    child_ordinal += 1
                    current_sentences = [s]
                    current_words = s_words
                else:
                    current_sentences.append(s)
                    current_words.extend(s_words)

            if current_sentences:
                yield emit_child(current_sentences, child_ordinal)
                child_ordinal += 1


def chunk_documents(documents, chunker: Chunker) -> Iterator[Chunk]:
    for document in documents:
        yield from chunker.chunk(document)

