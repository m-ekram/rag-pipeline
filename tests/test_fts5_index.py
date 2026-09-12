import pytest
from ingestion.documents import Chunk
from retrieval.fts5_index import FTS5Index


def test_fts5_index_basic_and_slashes():
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="Voter with EPIC BR/35/207/291052 named Mohammad Maiyyar age 57",
            ordinal=0,
            metadata={"page": 29},
        ),
        Chunk(
            chunk_id="c2",
            doc_id="d1",
            text="Voter with EPIC BR/35/207/291335 named Faisal Alam age 38",
            ordinal=1,
            metadata={"page": 17},
        ),
        Chunk(
            chunk_id="c3",
            doc_id="d2",
            text="Information on page 66 regarding Table 24 URDPFI guidelines",
            ordinal=0,
            metadata={"page": 66},
        ),
    ]

    index = FTS5Index(":memory:").build(chunks)

    # Test exact slash match
    results = index.search("Who has voter ID BR/35/207/291052?")
    assert len(results) > 0
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0

    # Test page filter
    results_p66 = index.search("guidelines", page_num=66)
    assert len(results_p66) == 1
    assert results_p66[0].chunk_id == "c3"

    # Test get_by_page
    p29_chunks = index.get_by_page(29)
    assert len(p29_chunks) == 1
    assert p29_chunks[0].chunk_id == "c1"


def test_search_field_falls_back_to_substring_match_without_crashing():
    """OCR glues words ("MdZahidKhan"), so FTS has no matching token. The LIKE
    fallback used to call .strip() on a term group and crash the whole answer."""
    index = FTS5Index(":memory:").build([
        Chunk(chunk_id="v1", doc_id="d1", ordinal=0,
              text="- [Serial: 5 | Voter: Ali | Relation: Father: MdZahidKhan]"),
        Chunk(chunk_id="v2", doc_id="d1", ordinal=1,
              text="- [Serial: 6 | Voter: Ravi | Relation: Father: Ram Khan]"),
    ])
    hits = index.search_field("Relation", [["Zahid", "जाहिद"], ["Khan", "खान"]])
    # Every group is required: "Ram Khan" satisfies only one of them.
    assert [h.chunk_id for h in hits] == ["v1"]


def test_search_treats_operator_words_and_slashes_as_plain_terms():
    index = FTS5Index(":memory:").build([
        Chunk(chunk_id="c1", doc_id="d1", ordinal=0,
              text="Voter with EPIC BR/35/207/291052 lives in house 4"),
    ])
    # Bare NOT / AND are FTS5 operators; unquoted they raised a syntax error
    # that was swallowed into zero results.
    assert [h.chunk_id for h in index.search("voters NOT in house 4")] == ["c1"]
    assert [h.chunk_id for h in index.search("house AND 4")] == ["c1"]
    # Prefix query on a slash ID: the router's fallback when an exact ID misses.
    assert [h.chunk_id for h in index.search("BR/35/207/29105*")] == ["c1"]


def test_stored_payload_does_not_duplicate_chunk_text():
    index = FTS5Index(":memory:").build([
        Chunk(chunk_id="c1", doc_id="d1", ordinal=0, text="only once", metadata={"page": 3}),
    ])
    (meta_json,) = index.con.execute("SELECT metadata_json FROM chunks_fts").fetchone()
    assert "only once" not in meta_json
    (chunk,) = index.get_by_page(3)
    assert chunk.text == "only once"
