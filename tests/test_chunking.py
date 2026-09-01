"""Chunking: the contextual header, the size floor, and id stability.

These are the properties the retrieval numbers depend on, so they are worth
pinning: a silently dropped header or a changed chunk_id invalidates both the
embedding cache and every measured result.
"""

from langchain_core.documents import Document

import config
from app.chunking import _chunk_id, _header, naive_split, stats, structured_split


def doc(text, **meta):
    meta.setdefault("source", "guide.md")
    meta.setdefault("title", "Guide")
    return Document(page_content=text, metadata=meta)


# -- contextual headers --------------------------------------------------------


def test_structured_split_prepends_a_contextual_header():
    chunks = structured_split([doc("The timeout is 30s. " * 20)])
    assert chunks
    assert all(c.page_content.startswith("[Guide]") for c in chunks)


def test_header_includes_the_page_number_when_there_is_one():
    assert _header({"title": "Networking", "page": 12}) == "[Networking - p.12]"
    assert _header({"title": "Networking"}) == "[Networking]"


def test_header_falls_back_to_source_when_there_is_no_title():
    assert _header({"source": "manuals/net.pdf"}) == "[manuals/net.pdf]"


def test_headers_can_be_switched_off():
    chunks = structured_split([doc("The timeout is 30s. " * 20)], add_headers=False)
    assert chunks
    assert not any(c.page_content.startswith("[") for c in chunks)


def test_naive_split_never_adds_headers():
    """The baseline must stay header-free or the comparison measures nothing."""
    chunks = naive_split([doc("word " * 500)])
    assert chunks
    assert not any(c.page_content.startswith("[Guide]") for c in chunks)


# -- the size floor ------------------------------------------------------------


def test_fragments_below_the_floor_are_dropped():
    chunks = structured_split([doc("tiny")])
    assert chunks == []


def test_a_document_at_the_floor_survives():
    text = "x" * config.MIN_CHUNK_CHARS
    assert len(structured_split([doc(text)])) == 1


# -- metadata ------------------------------------------------------------------


def test_each_chunk_carries_index_id_and_char_count():
    chunks = structured_split([doc("The timeout is 30 seconds. " * 40)])
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert len(chunk.metadata["chunk_id"]) == 16
        assert chunk.metadata["char_count"] > 0


def test_chunk_index_restarts_for_each_source():
    long_text = "The timeout is 30 seconds. " * 40
    chunks = structured_split(
        [doc(long_text, source="a.md"), doc(long_text, source="b.md")]
    )
    first = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "a.md"]
    second = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "b.md"]
    assert first[0] == 0 and second[0] == 0


def test_chunk_id_is_stable_for_the_same_input():
    assert _chunk_id("hello", "a.md", 0) == _chunk_id("hello", "a.md", 0)


def test_chunk_id_changes_with_text_source_or_position():
    base = _chunk_id("hello", "a.md", 0)
    assert _chunk_id("goodbye", "a.md", 0) != base
    assert _chunk_id("hello", "b.md", 0) != base
    assert _chunk_id("hello", "a.md", 1) != base


def test_char_count_excludes_the_header():
    """char_count measures the source text, not the header we added to it."""
    chunks = structured_split([doc("The timeout is 30 seconds. " * 10)])
    chunk = chunks[0]
    assert chunk.metadata["char_count"] < len(chunk.page_content)


# -- splitter behaviour --------------------------------------------------------


def test_structured_split_respects_the_size_limit():
    chunks = structured_split([doc("The timeout is 30 seconds. " * 200)], chunk_size=300)
    assert all(c.metadata["char_count"] <= 300 for c in chunks)


def test_structured_split_prefers_paragraph_boundaries():
    """Two clean paragraphs that each fit should not be cut mid-sentence."""
    para = "Alpha sentence about configuration values and their defaults. " * 3
    other = "Beta sentence about deployment topology and its tradeoffs. " * 3
    chunks = structured_split([doc(f"{para}\n\n{other}")], chunk_size=250, chunk_overlap=0)
    assert len(chunks) >= 2


def test_stats_on_an_empty_list():
    assert stats([]) == {"count": 0}


def test_stats_reports_count_and_sources():
    long_text = "The timeout is 30 seconds. " * 40
    chunks = structured_split(
        [doc(long_text, source="a.md"), doc(long_text, source="b.md")]
    )
    s = stats(chunks)
    assert s["count"] == len(chunks)
    assert s["sources"] == 2
    assert s["min_chars"] <= s["median_chars"] <= s["max_chars"]
