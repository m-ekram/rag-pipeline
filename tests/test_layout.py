from pathlib import Path
import fitz
import pytest

from ingestion.layout import DigitalLayoutExtractor, table_to_markdown


def test_table_to_markdown_basic():
    grid = [
        ["Name", "Role", "Score"],
        ["Alice", "Admin", "95"],
        ["Bob", "User", "80"],
    ]
    md = table_to_markdown(grid)
    assert "| Name | Role | Score |" in md
    assert "| Alice | Admin | 95 |" in md


def test_pmp_report_page66_table_extraction():
    pmp_path = Path("data/pmp-2031-report.pdf")
    if not pmp_path.exists():
        pytest.skip("data/pmp-2031-report.pdf not found in workspace")

    doc = fitz.open(str(pmp_path))
    # Printed page 66 is at page index 79
    p66 = doc[79]

    extractor = DigitalLayoutExtractor()
    blocks = extractor.extract_page(p66, page_num=66)

    table_blocks = [b for b in blocks if b.type == "table"]
    assert len(table_blocks) >= 1

    # Verify that Table 24 / 25 contents are in Markdown format
    table_texts = "\n".join(b.content for b in table_blocks)
    assert "|" in table_texts
    # Check for proposed residential land use in Table 25
    assert "Residential" in table_texts or "URDPFI" in table_texts


def test_pmp_report_page87_table26_extraction():
    pmp_path = Path("data/pmp-2031-report.pdf")
    if not pmp_path.exists():
        pytest.skip("data/pmp-2031-report.pdf not found in workspace")

    doc = fitz.open(str(pmp_path))
    # Page 87 (printed page 73) is at index 86
    p87 = doc[86]

    extractor = DigitalLayoutExtractor()
    blocks = extractor.extract_page(p87, page_num=87)

    table_blocks = [b for b in blocks if b.type == "table"]
    assert len(table_blocks) == 1

    tbl = table_blocks[0]
    # Check that Table 26 caption was bound directly to the table
    assert tbl.metadata.get("caption") == "Table 26: Existing Width of Roads"
    assert "### Table 26: Existing Width of Roads" in tbl.content
    # Check that multi-tier header was consolidated and split columns reconciled
    assert "Rajendra Path" in tbl.content
    assert "26.21 / 10.97" in tbl.content
    assert "16.76" in tbl.content
    assert "16.62" in tbl.content
    assert "Avg. Width" in tbl.content
    assert "Metalled Width" in tbl.content


def test_pmp_report_page131_table43_extraction():
    pmp_path = Path("data/pmp-2031-report.pdf")
    if not pmp_path.exists():
        pytest.skip("data/pmp-2031-report.pdf not found in workspace")

    doc = fitz.open(str(pmp_path))
    # Page 131 is at index 130
    p131 = doc[130]

    extractor = DigitalLayoutExtractor()
    blocks = extractor.extract_page(p131, page_num=131)

    table_blocks = [b for b in blocks if b.type == "table"]
    assert len(table_blocks) == 1

    tbl = table_blocks[0]
    assert tbl.metadata.get("caption") == "Table 43: Primary Health Sub-Centres" or "Table 43" in tbl.metadata.get("caption", "")
    assert "### Table 43: Primary Health Sub-Centres" in tbl.content
    assert "Phulwari Sharif" in tbl.content
    assert "Required 2031" in tbl.content
    assert "Maner" in tbl.content


def test_stitch_continuation_tables_dicts():
    from ingestion.layout import stitch_continuation_tables

    page1 = {
        "page_num": 1,
        "blocks": [
            {
                "block_type": "table",
                "caption": "Table 10: Road Inventory",
                "headers": ["No.", "Road Name", "Width"],
                "rows": [["1", "Station Road", "15m"], ["2", "Circular Road", "20m"]],
            }
        ],
    }

    page2 = {
        "page_num": 2,
        "blocks": [
            {
                "block_type": "table",
                "caption": "",
                "headers": ["No.", "Road Name", "Width"],
                "rows": [["3", "Bailey Road", "30m"], ["4", "Boring Road", "25m"]],
            }
        ],
    }

    stitched = stitch_continuation_tables([page1, page2])
    assert len(stitched) == 2

    # Check page 2 table inherited caption and continuation metadata
    p2_tbl = stitched[1]["blocks"][0]
    assert p2_tbl["is_continuation"] is True
    assert p2_tbl["caption"] == "Table 10: Road Inventory (Continued)"
    assert "Station Road" in p2_tbl["parent_text"]
    assert "Bailey Road" in p2_tbl["parent_text"]
    assert "Rows 1-4" in p2_tbl["parent_text"]


def test_stitch_continuation_tables_documents():
    from ingestion.documents import Document
    from ingestion.layout import stitch_continuation_tables

    doc1_text = """### Table 26: Existing Width of Roads
| No. | Name of the road | Max. Width |
| --- | --- | --- |
| 1 | Mainpura Road | 8.84 |
| 20 | Bari Path | 21.95 |
"""
    doc2_text = """| No. | Name of the road | Max. Width |
| --- | --- | --- |
| 21 | Ashok Rajpath | 25.00 |
| 46 | Bailey Road | 87.17 |
"""

    doc1 = Document(doc_id="report#p87", text=doc1_text, page=87)
    doc2 = Document(doc_id="report#p88", text=doc2_text, page=88)

    stitched_docs = stitch_continuation_tables([doc1, doc2])
    assert len(stitched_docs) == 2

    # Page 88 must now have continuation caption
    assert "### Table 26: Existing Width of Roads (Continued)" in stitched_docs[1].text
    assert stitched_docs[1].metadata.get("is_continuation") is True

    # Check unified parent text contains all rows
    stitched_meta = stitched_docs[1].metadata.get("stitched_tables", {})
    assert "Table 26: Existing Width of Roads (Continued)" in stitched_meta
    parent_text = stitched_meta["Table 26: Existing Width of Roads (Continued)"]["parent_text"]
    assert "Mainpura Road" in parent_text
    assert "Bailey Road" in parent_text
    assert "Rows 1-4" in parent_text



