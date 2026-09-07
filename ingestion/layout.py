"""Layout compilation and structure recovery for documents.

Represents documents as typed LayoutBlocks (table, text, title, header, footer)
and provides digital native table extraction into clean Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence
import re


@dataclass
class LayoutBlock:
    """A single layout element on a document page."""

    type: str  # 'title', 'text', 'table', 'figure', 'header', 'footer'
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    content: str  # Plain text OR clean Markdown Table string (`| col1 | col2 |`)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_table(self) -> bool:
        return self.type == "table"


def bind_table_captions(blocks: list, tables: list, max_gap_pts: float = 40.0):
    """Find text blocks immediately above a table acting as captions and bind them to the table."""
    caption_regex = re.compile(r"^(?:Table|Tab\.?|Annexure|Schedule)\s*\d+[:.\s-]", re.IGNORECASE)
    used_block_indices = set()
    for table in tables:
        t_bbox = getattr(table, "bbox", None) or (table.get("bbox") if isinstance(table, dict) else None)
        if not t_bbox:
            continue
        best_block = None
        best_idx = None
        min_dist = max_gap_pts
        for idx, blk in enumerate(blocks):
            if idx in used_block_indices:
                continue
            if isinstance(blk, dict):
                b_bbox = blk.get("bbox")
                b_text = blk.get("text", "").strip()
            elif isinstance(blk, LayoutBlock):
                b_bbox = blk.bbox
                b_text = blk.content.strip()
            elif isinstance(blk, (list, tuple)) and len(blk) >= 5:
                b_bbox = blk[:4]
                b_text = str(blk[4]).strip()
            else:
                continue

            if b_bbox and b_bbox[3] <= t_bbox[1]:
                vertical_gap = t_bbox[1] - b_bbox[3]
                if vertical_gap <= min_dist:
                    if caption_regex.search(b_text) or len(b_text.splitlines()) <= 2:
                        min_dist = vertical_gap
                        best_block = blk
                        best_idx = idx
        if best_block:
            used_block_indices.add(best_idx)
            if isinstance(best_block, dict):
                cap_text = best_block.get("text", "").strip()
            elif isinstance(best_block, LayoutBlock):
                cap_text = best_block.content.strip()
            else:
                cap_text = str(best_block[4]).strip()
            setattr(table, "caption", cap_text)
        else:
            if not hasattr(table, "caption"):
                setattr(table, "caption", "")

    remaining_blocks = [blk for idx, blk in enumerate(blocks) if idx not in used_block_indices]
    return remaining_blocks, tables


def table_to_markdown(grid: Sequence[Sequence[Optional[str]]], caption: str = "") -> str:
    """Convert a 2D grid into clean Markdown with multi-tier header consolidation and split column reconciliation."""
    if not grid or not any(grid):
        return ""

    # Clean cells: collapse newlines, strip whitespace, handle None, escape pipes
    cleaned_grid: list[list[str]] = []
    max_cols = 0
    for row in grid:
        cleaned_row = []
        for cell in row:
            val = str(cell or "").strip()
            val = re.sub(r"\s*\n\s*", " ", val)
            val = val.replace("|", "\\|")
            cleaned_row.append(val)
        cleaned_grid.append(cleaned_row)
        if len(cleaned_row) > max_cols:
            max_cols = len(cleaned_row)

    if not cleaned_grid or max_cols == 0:
        return ""

    for row in cleaned_grid:
        while len(row) < max_cols:
            row.append("")

    # 1. Identify where data rows start
    data_start_idx = 1
    for r_idx in range(len(cleaned_grid)):
        row = cleaned_grid[r_idx]
        first_non_empty = ""
        has_decimal_or_pct = False
        non_year_numbers = 0
        for cell in row:
            s = cell.strip()
            if s and not first_non_empty:
                first_non_empty = s
            if re.search(r"\d+\.\d+", s) or re.search(r"\b\d+%\b", s):
                has_decimal_or_pct = True
            clean_s = s.replace(",", "")
            if re.search(r"\d+", clean_s):
                # Ignore year numbers (e.g. 2001, 2031, 2016 - 21) when checking for data rows
                if not re.match(r"^(?:19|20)\d\d(?:\s*[-–]\s*(?:\d\d|\d{4}))?$", clean_s):
                    non_year_numbers += 1

        # Check serial number (1..999, not a 4-digit year)
        if re.match(r"^[1-9]\d{0,2}\.?$", first_non_empty):
            data_start_idx = r_idx
            break
        # Check if row contains real data (decimals, percentages, or multiple non-year counts)
        if r_idx > 0 and (has_decimal_or_pct or non_year_numbers >= 2):
            data_start_idx = r_idx
            break

    # 2. Consolidate rows 0 to data_start_idx into a single header
    header_rows = cleaned_grid[:data_start_idx]
    data_rows = cleaned_grid[data_start_idx:]
    if not data_rows:
        data_rows = cleaned_grid[1:]
        header_rows = cleaned_grid[:1]

    consolidated_headers: list[str] = []
    for c in range(max_cols):
        col_parts: list[str] = []
        for r in header_rows:
            if c < len(r) and r[c]:
                val = r[c].strip()
                if val and val not in col_parts:
                    col_parts.append(val)
        col_name = " ".join(col_parts).strip() or f"Col {c+1}"
        consolidated_headers.append(col_name)

    # 3. Reconcile adjacent split columns from PyMuPDF header cell drifting
    for c in range(max_cols - 1):
        has_data_c = any(c < len(r) and r[c].strip() for r in data_rows)
        has_data_next = any(c + 1 < len(r) and r[c + 1].strip() for r in data_rows)
        header_c = consolidated_headers[c]
        header_next = consolidated_headers[c + 1]
        if has_data_c and not has_data_next and header_c.startswith("Col ") and not header_next.startswith("Col "):
            consolidated_headers[c] = header_next
            consolidated_headers[c + 1] = ""
        elif not has_data_c and has_data_next and not header_c.startswith("Col ") and header_next.startswith("Col "):
            consolidated_headers[c + 1] = header_c
            consolidated_headers[c] = ""

    # 4. Filter columns that have data or a meaningful non-empty header
    keep_cols = [
        c for c in range(max_cols)
        if any(c < len(r) and r[c].strip() for r in data_rows) and consolidated_headers[c]
    ]
    if not keep_cols:
        keep_cols = [c for c in range(max_cols) if consolidated_headers[c]]

    clean_headers = [consolidated_headers[c] for c in keep_cols]

    # 5. Build Markdown
    lines = []
    if caption:
        lines.append(f"### {caption}")
    lines.append("| " + " | ".join(clean_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(clean_headers)) + " |")
    for r in data_rows:
        row_vals = [r[c].strip() if c < len(r) else "" for c in keep_cols]
        if any(row_vals):
            lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def _intersects(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> bool:
    """Check if two bounding boxes (x0, y0, x1, y1) have meaningful overlap."""
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return False
    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    return (overlap_area / max(b1_area, 1.0)) > 0.3


class DigitalLayoutExtractor:
    """Extracts tables and structured layout blocks from digital PDFs with zero OCR overhead."""

    def __init__(self, header_margin_pct: float = 0.06, footer_margin_pct: float = 0.06, max_caption_gap_pts: float = 40.0):
        self.header_margin_pct = header_margin_pct
        self.footer_margin_pct = footer_margin_pct
        self.max_caption_gap_pts = max_caption_gap_pts

    def extract_page(self, page, page_num: int) -> list[LayoutBlock]:
        """Extract layout blocks (tables and text) in vertical reading order from a fitz.Page."""
        blocks: list[LayoutBlock] = []
        page_rect = page.rect
        page_height = page_rect.height
        header_threshold = page_height * self.header_margin_pct
        footer_threshold = page_height * (1.0 - self.footer_margin_pct)

        table_bboxes: list[tuple[float, float, float, float]] = []
        raw_tables = []

        # 1. Detect native tables
        try:
            tab_finder = page.find_tables()
            if tab_finder and tab_finder.tables:
                raw_tables = list(tab_finder.tables)
        except Exception:
            raw_tables = []

        # 2. Extract text blocks and bind table captions
        raw_text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        caption_regex = re.compile(r"^(?:Table|Tab\.?|Annexure|Schedule)\s*\d+[:.\s-]", re.IGNORECASE)
        table_captions: dict[int, str] = {}
        used_tb_indices: set[int] = set()

        for t_idx, tab in enumerate(raw_tables):
            t_bbox = tuple(float(c) for c in tab.bbox)
            best_idx = None
            min_dist = self.max_caption_gap_pts
            for tb_idx, tb in enumerate(raw_text_blocks):
                if tb_idx in used_tb_indices or len(tb) < 5:
                    continue
                if len(tb) >= 7 and tb[6] != 0:
                    continue
                bx0, by0, bx1, by1, btext = tb[0], tb[1], tb[2], tb[3], tb[4].strip()
                if not btext:
                    continue
                if by1 <= t_bbox[1]:
                    vertical_gap = t_bbox[1] - by1
                    if vertical_gap <= min_dist:
                        if caption_regex.search(btext) or len(btext.splitlines()) <= 2:
                            min_dist = vertical_gap
                            best_idx = tb_idx

            if best_idx is not None:
                used_tb_indices.add(best_idx)
                table_captions[t_idx] = raw_text_blocks[best_idx][4].strip()

        # 3. Convert tables to Markdown with captions bound
        for t_idx, tab in enumerate(raw_tables):
            raw_data = tab.extract()
            caption = table_captions.get(t_idx, "")
            md_table = table_to_markdown(raw_data, caption=caption)
            if md_table:
                bbox = tuple(float(c) for c in tab.bbox)
                table_bboxes.append(bbox)
                blocks.append(
                    LayoutBlock(
                        type="table",
                        bbox=bbox,
                        content=md_table,
                        metadata={
                            "page": page_num,
                            "table_index": t_idx + 1,
                            "caption": caption,
                            "rows": len(raw_data),
                            "cols": len(raw_data[0]) if raw_data else 0,
                        },
                    )
                )

        # 4. Extract non-table, non-caption text blocks
        for tb_idx, tb in enumerate(raw_text_blocks):
            if tb_idx in used_tb_indices or len(tb) < 5:
                continue
            x0, y0, x1, y1, text = tb[0], tb[1], tb[2], tb[3], tb[4]
            cleaned_text = text.strip()
            if not cleaned_text:
                continue

            bbox = (float(x0), float(y0), float(x1), float(y1))

            # Skip text blocks that fall inside extracted tables
            if any(_intersects(bbox, tbox) for tbox in table_bboxes):
                continue

            # Classify running headers / footers
            block_type = "text"
            if y1 <= header_threshold:
                block_type = "header"
            elif y0 >= footer_threshold:
                block_type = "footer"
            elif len(cleaned_text.splitlines()) == 1 and len(cleaned_text) < 80:
                if cleaned_text.isupper() or cleaned_text.startswith(("#", "Table", "Chapter", "Section")):
                    block_type = "title"

            blocks.append(
                LayoutBlock(
                    type=block_type,
                    bbox=bbox,
                    content=cleaned_text,
                    metadata={"page": page_num},
                )
            )

        # 5. Topologically sort in natural top-to-bottom, left-to-right reading order
        blocks.sort(key=lambda b: (round(b.bbox[1] / 15.0) * 15.0, b.bbox[0]))
        return blocks


def parse_markdown_tables(text: str) -> list[dict[str, Any]]:
    """Extract Markdown tables and their preceding captions from page text."""
    caption_regex = re.compile(r"^(?:Table|Tab\.?|Annexure|Schedule)\s*\d+[:.\s-]", re.IGNORECASE)
    lines = text.splitlines()
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i]
        caption = ""
        trimmed = line.strip()
        if trimmed.startswith("#") or caption_regex.search(trimmed):
            next_k = i + 1
            while next_k < len(lines) and not lines[next_k].strip():
                next_k += 1
            if next_k + 1 < len(lines) and lines[next_k].strip().startswith("|") and re.match(r"^\s*\|\s*[-:]+", lines[next_k + 1]):
                caption = trimmed.lstrip("#").strip()
                i = next_k
                line = lines[i]

        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|\s*[-:]+", lines[i + 1]):
            start_line_idx = i
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1

            raw_headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
            rows = []
            for r in table_lines[2:]:
                cells = [c.strip() for c in r.split("|")[1:-1]]
                if any(cells):
                    rows.append(cells)
            tables.append({
                "caption": caption,
                "headers": raw_headers,
                "rows": rows,
                "start_line": start_line_idx,
                "end_line": i,
                "table_lines": table_lines,
            })
            continue
        i += 1
    return tables


def stitch_continuation_tables(
    extracted_pages: list[Any],
    column_similarity_threshold: float = 0.7,
) -> list[Any]:
    """Scans sequential pages and merges tables that span across page boundaries.

    A table on Page N+1 is considered a continuation of a table on Page N if:
    1. Page N ends with a table (or table is near the bottom).
    2. Page N+1 begins with a table near the top.
    3. The column count and header names match closely.
    4. Row indices follow a continuous sequence (e.g. Page N ends at Row 20, Page N+1 starts at Row 21).

    Supports:
    - Lists of page dicts containing layout blocks ({"blocks": [...], "page_num": N})
    - Lists of Document instances containing Markdown text and metadata.
    """
    if not extracted_pages or len(extracted_pages) < 2:
        return extracted_pages

    first_item = extracted_pages[0]
    is_doc_list = hasattr(first_item, "text") and hasattr(first_item, "doc_id")

    if is_doc_list:
        return _stitch_continuation_tables_documents(extracted_pages, column_similarity_threshold)
    else:
        return _stitch_continuation_tables_dicts(extracted_pages, column_similarity_threshold)


def _stitch_continuation_tables_dicts(
    extracted_pages: list[dict[str, Any]],
    column_similarity_threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """Stitch continuation tables across page dictionary / LayoutBlock structures."""
    for i in range(len(extracted_pages) - 1):
        prev_page = extracted_pages[i]
        curr_page = extracted_pages[i + 1]

        prev_tables = [
            b for b in prev_page.get("blocks", [])
            if (isinstance(b, dict) and b.get("block_type") == "table")
            or (hasattr(b, "type") and b.type == "table")
        ]
        curr_tables = [
            b for b in curr_page.get("blocks", [])
            if (isinstance(b, dict) and b.get("block_type") == "table")
            or (hasattr(b, "type") and b.type == "table")
        ]

        if not prev_tables or not curr_tables:
            continue

        last_table_prev = prev_tables[-1]
        first_table_curr = curr_tables[0]

        def get_field(tb, key, default=None):
            if isinstance(tb, dict):
                return tb.get(key, default)
            meta = getattr(tb, "metadata", {}) or {}
            if key in meta:
                return meta[key]
            return getattr(tb, key, default)

        def set_field(tb, key, val):
            if isinstance(tb, dict):
                tb[key] = val
            else:
                if hasattr(tb, "metadata") and isinstance(tb.metadata, dict):
                    tb.metadata[key] = val
                setattr(tb, key, val)

        prev_headers = get_field(last_table_prev, "headers", [])
        curr_headers = get_field(first_table_curr, "headers", [])

        headers_match = False
        if prev_headers and curr_headers and len(prev_headers) == len(curr_headers):
            matches = sum(
                1 for p, c in zip(prev_headers, curr_headers)
                if str(p).lower() == str(c).lower() or "col" in str(c).lower() or "col" in str(p).lower()
            )
            if (matches / len(prev_headers)) >= column_similarity_threshold:
                headers_match = True

        prev_rows = get_field(last_table_prev, "rows", [])
        curr_rows = get_field(first_table_curr, "rows", [])

        is_continuation = False
        if headers_match:
            is_continuation = True
        elif prev_rows and curr_rows:
            first_cell_prev_last = str(prev_rows[-1][0] or "").strip().rstrip(".")
            first_cell_curr_first = str(curr_rows[0][0] or "").strip().rstrip(".")
            if first_cell_prev_last.isdigit() and first_cell_curr_first.isdigit():
                if int(first_cell_curr_first) == int(first_cell_prev_last) + 1:
                    is_continuation = True

        if is_continuation:
            prev_page_num = prev_page.get("page_num", i + 1)
            parent_caption = get_field(last_table_prev, "caption") or f"Table (Page {prev_page_num})"
            clean_caption = re.sub(r"[^\w]+", "_", parent_caption).strip("_")
            unified_table_id = get_field(last_table_prev, "table_id") or f"tbl_stitched_{clean_caption}_p{prev_page_num}"

            combined_rows = prev_rows + curr_rows
            clean_headers = prev_headers or curr_headers

            unified_markdown = (
                f"### {parent_caption} (Complete / Continuous Rows 1-{len(combined_rows)})\n"
                f"| {' | '.join(clean_headers)} |\n"
                f"| {' | '.join(['---'] * len(clean_headers))} |\n"
            )
            for r in combined_rows:
                unified_markdown += f"| {' | '.join([str(c or '').replace(chr(10), ' ') for c in r])} |\n"

            set_field(last_table_prev, "stitched_with_next", True)
            set_field(last_table_prev, "parent_text", unified_markdown)
            set_field(last_table_prev, "table_id", unified_table_id)

            set_field(first_table_curr, "is_continuation", True)
            set_field(first_table_curr, "caption", f"{parent_caption} (Continued)")
            set_field(first_table_curr, "parent_text", unified_markdown)
            set_field(first_table_curr, "parent_id", unified_table_id)
            set_field(first_table_curr, "headers", clean_headers)

    return extracted_pages


def _stitch_continuation_tables_documents(
    docs: list[Any],
    column_similarity_threshold: float = 0.7,
) -> list[Any]:
    """Stitch continuation tables across Document units (Markdown text & metadata)."""
    from .documents import Document

    updated_docs = list(docs)

    for i in range(len(updated_docs) - 1):
        prev_doc = updated_docs[i]
        curr_doc = updated_docs[i + 1]

        prev_tables = parse_markdown_tables(prev_doc.text)
        curr_tables = parse_markdown_tables(curr_doc.text)

        if not prev_tables or not curr_tables:
            continue

        last_table_prev = prev_tables[-1]
        first_table_curr = curr_tables[0]

        prev_headers = last_table_prev.get("headers", [])
        curr_headers = first_table_curr.get("headers", [])

        headers_match = False
        if prev_headers and curr_headers and len(prev_headers) == len(curr_headers):
            matches = sum(
                1 for p, c in zip(prev_headers, curr_headers)
                if str(p).lower() == str(c).lower() or "col" in str(c).lower() or "col" in str(p).lower()
            )
            if (matches / len(prev_headers)) >= column_similarity_threshold:
                headers_match = True

        prev_rows = last_table_prev.get("rows", [])
        curr_rows = first_table_curr.get("rows", [])

        is_continuation = False
        if headers_match:
            is_continuation = True
        elif prev_rows and curr_rows:
            first_cell_prev_last = str(prev_rows[-1][0] or "").strip().rstrip(".")
            first_cell_curr_first = str(curr_rows[0][0] or "").strip().rstrip(".")
            if first_cell_prev_last.isdigit() and first_cell_curr_first.isdigit():
                if int(first_cell_curr_first) == int(first_cell_prev_last) + 1:
                    is_continuation = True

        if is_continuation:
            parent_caption = (
                last_table_prev.get("caption")
                or prev_doc.metadata.get("stitched_caption")
                or f"Table (Page {prev_doc.page})"
            )
            clean_caption = re.sub(r"[^\w]+", "_", parent_caption).strip("_")
            unified_table_id = f"{prev_doc.doc_id}#table_{clean_caption}_stitched"

            clean_headers = prev_headers or curr_headers
            combined_rows = prev_rows + curr_rows

            unified_markdown = (
                f"### {parent_caption} (Complete / Continuous Rows 1-{len(combined_rows)})\n"
                f"| {' | '.join(clean_headers)} |\n"
                f"| {' | '.join(['---'] * len(clean_headers))} |\n"
            )
            for r in combined_rows:
                unified_markdown += f"| {' | '.join([str(c or '').replace(chr(10), ' ') for c in r])} |\n"

            # Insert caption into Page N+1 text if it doesn't already have one
            curr_lines = curr_doc.text.splitlines()
            start_idx = first_table_curr["start_line"]
            has_existing_caption = False
            if start_idx > 0 and curr_lines[start_idx - 1].strip().startswith("###"):
                has_existing_caption = True

            if not has_existing_caption:
                curr_lines.insert(start_idx, f"### {parent_caption} (Continued)")
                new_curr_text = "\n".join(curr_lines)
            else:
                new_curr_text = curr_doc.text

            prev_stitched = dict(prev_doc.metadata.get("stitched_tables", {}))
            prev_stitched[parent_caption] = {
                "parent_id": unified_table_id,
                "parent_text": unified_markdown,
                "caption": parent_caption,
            }
            updated_docs[i] = Document(
                doc_id=prev_doc.doc_id,
                text=prev_doc.text,
                title=prev_doc.title,
                source=prev_doc.source,
                page=prev_doc.page,
                section=prev_doc.section,
                metadata={**prev_doc.metadata, "stitched_tables": prev_stitched},
            )

            curr_stitched = dict(curr_doc.metadata.get("stitched_tables", {}))
            curr_stitched[f"{parent_caption} (Continued)"] = {
                "parent_id": unified_table_id,
                "parent_text": unified_markdown,
                "caption": f"{parent_caption} (Continued)",
            }
            curr_stitched[parent_caption] = curr_stitched[f"{parent_caption} (Continued)"]

            updated_docs[i + 1] = Document(
                doc_id=curr_doc.doc_id,
                text=new_curr_text,
                title=curr_doc.title,
                source=curr_doc.source,
                page=curr_doc.page,
                section=curr_doc.section or parent_caption,
                metadata={
                    **curr_doc.metadata,
                    "stitched_tables": curr_stitched,
                    "stitched_caption": parent_caption,
                    "is_continuation": True,
                },
            )

    return updated_docs


def linearize_table_row(headers: list[str], row_cells: list[str]) -> str:
    """Convert a table row into a natural record sentence for high cross-encoder scoring."""
    pairs = []
    for h, c in zip(headers, row_cells):
        h_clean = h.strip().replace("\n", " ")
        c_clean = c.strip().replace("\n", " ")
        if h_clean and c_clean:
            pairs.append(f"{h_clean}: {c_clean}")
    return "[Record: " + " | ".join(pairs) + "]"
