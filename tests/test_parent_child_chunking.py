from ingestion.chunking import StructureAwareParentChildChunker
from ingestion.documents import Document


def test_parent_child_table_chunking():
    md_table = """| Sl. No. | Land Use Categories | URDPFI (%) | Proposed (%) |
| --- | --- | --- | --- |
| 1 | Residential | 42 | 55.04 |
| 2 | Commercial | 5 | 7.20 |
| 3 | Industrial | 8 | 6.10 |
| 4 | Public/Semi-Public | 12 | 14.50 |
"""
    doc = Document(
        doc_id="pmp_p66",
        text=md_table,
        page=66,
    )

    chunker = StructureAwareParentChildChunker(rows_per_child=2)
    chunks = list(chunker.chunk(doc))

    assert len(chunks) == 2
    for c in chunks:
        # Every child chunk must have the table header prepended
        assert "| Sl. No. | Land Use Categories |" in c.text
        # Every child chunk must have parent metadata linking back to the whole table
        assert c.metadata.get("block_type") == "table"
        assert c.metadata.get("parent_id") is not None
        assert "Residential" in c.metadata.get("parent_text")
        assert "Public/Semi-Public" in c.metadata.get("parent_text")


def test_parent_child_text_chunking():
    text = (
        "This is paragraph one about urban planning norms adopted in Patna. "
        "It describes civic facilities for the projected population of the city.\n\n"
        "This is paragraph two discussing residential densities across various zones. "
        "The proposed master plan allocates land according to national standards."
    )
    doc = Document(
        doc_id="pmp_p52",
        text=text,
        page=52,
    )

    chunker = StructureAwareParentChildChunker(text_child_words=20)
    chunks = list(chunker.chunk(doc))
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata.get("block_type") == "text"
        assert c.metadata.get("parent_text") == text


def test_parent_child_table_chunking_with_caption():
    text = """The form of road network is influenced by geography.

### Table 26: Existing Width of Roads
| No. | Name of the road | Max. /Min Width(in M) | Avg. Width (in M) |
| --- | --- | --- | --- |
| 1 | Rajendra Path | 26.21 / 10.97 | 16.76 |
| 2 | Barj Kishore Path | 23.54/ 17.06 | 20.11 |
| 3 | Mazharul Haque Path | 32.16/ 22.66 | 27.0 |
"""
    doc = Document(
        doc_id="pmp_p87",
        text=text,
        page=87,
    )

    chunker = StructureAwareParentChildChunker(rows_per_child=1)
    chunks = list(chunker.chunk(doc))

    table_chunks = [c for c in chunks if c.metadata.get("block_type") == "table"]
    assert len(table_chunks) == 3

    # Every child chunk must have the caption prepended
    for c in table_chunks:
        assert "### Table 26: Existing Width of Roads" in c.text
        assert "| No. | Name of the road |" in c.text
        assert c.metadata.get("caption") == "Table 26: Existing Width of Roads"
        assert "Rajendra Path" in c.metadata.get("parent_text")
        assert "Mazharul Haque Path" in c.metadata.get("parent_text")

    # The Rajendra Path child chunk has all the key terms
    rajendra_chunk = table_chunks[0]
    assert "Rajendra Path" in rajendra_chunk.text
    assert "16.76" in rajendra_chunk.text


def test_masterclass_upgrades():
    from ingestion.layout import linearize_table_row
    from ingestion.chunking import canonicalize_house_number, dual_house_number_display
    from generation.prompts import SYSTEM_PROMPT

    # Upgrade 1: Citation guard
    assert "CRITICAL CITATION RULES:" in SYSTEM_PROMPT
    assert "NEVER output original paper reference numbers" in SYSTEM_PROMPT

    # Upgrade 2: Table row linearization
    headers = ["Architecture", "Modality", "Metrics"]
    cells = ["YOLOv8", "Cephalometric", "SDR 2mm: 86.31%"]
    linearized = linearize_table_row(headers, cells)
    assert linearized == "[Record: Architecture: YOLOv8 | Modality: Cephalometric | Metrics: SDR 2mm: 86.31%]"

    # Upgrade 3: Devanagari digit normalizer
    assert canonicalize_house_number("३६") == "36"
    assert canonicalize_house_number("36") == "36"
    assert dual_house_number_display("36") == "36 / ३६"
    assert dual_house_number_display("३६") == "36 / ३६"


