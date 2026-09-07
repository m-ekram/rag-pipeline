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
