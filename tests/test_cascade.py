"""Tests for lexical-first retrieval and electoral record chunking.

The property under test throughout is that abstention survives. Dense retrieval
returns something for every query, so any path that consults it unconditionally
destroys the cheapest abstention signal the system has.
"""

import pytest

from ingestion.documents import Chunk, Document
from ingestion.electoral_chunking import (
    ElectoralRecordChunker,
    extract_epics,
    extract_header,
)
from retrieval.cascade import LexicalFirstRetriever
from retrieval.types import ScoredChunk

HEADER = ("विधानसभा निर्वाचन क्षेत्र की संख्या एवं नाम: 183-कुम्हरार\n"
          "भाग संख्या:: 1\n"
          "अनुभाग संख्या एवं नाम: 1-गोविन्द मित्रा रोड\n")

PAGE = HEADER + (
    "151]\nSHS5213004\n152]\nSHS0796649\n"
    "निर्वाचक का नाम: राजेश कुमार\nनिर्वाचक का नाम: मीना कुमारी\n"
    "मकान संख्या: 27\nमकान संख्या: 28\n"
    "153]\nSHS0582973\n154\nBR/35/207/267110\n"
    "निर्वाचक का नाम: सुजीत कुमार\nनिर्वाचक का नाम: अनिता देवी\n"
)


def _doc(text=PAGE):
    return Document(doc_id="roll#p8", text=text, title="Roll", source="/roll.pdf", page=8)


# --- header / EPIC extraction -------------------------------------------


def test_header_captures_constituency_part_and_section():
    header = extract_header(PAGE)
    assert "183-कुम्हरार" in header
    assert "भाग संख्या" in header
    assert "गोविन्द मित्रा रोड" in header


def test_header_is_empty_when_absent():
    assert extract_header("निर्वाचक का नाम: राजेश कुमार") == ""


def test_epic_extraction_handles_both_formats():
    epics = extract_epics(PAGE)
    assert "SHS5213004" in epics
    assert "BR/35/207/267110" in epics


# --- record chunking ----------------------------------------------------


def test_every_chunk_carries_page_context():
    """A bare list of names has no constituency, which is why fixed-size
    windows looked alike to the embedder and produced unusable citations."""
    chunks = list(ElectoralRecordChunker(bands_per_chunk=1).chunk(_doc()))
    assert chunks
    for chunk in chunks:
        assert "183-कुम्हरार" in chunk.text
        assert chunk.section and "कुम्हरार" in chunk.section


def test_chunks_split_on_record_bands_not_word_counts():
    one = list(ElectoralRecordChunker(bands_per_chunk=1).chunk(_doc()))
    two = list(ElectoralRecordChunker(bands_per_chunk=2).chunk(_doc()))
    assert len(one) > len(two)


def test_epics_are_recorded_as_metadata():
    chunks = list(ElectoralRecordChunker(bands_per_chunk=8).chunk(_doc()))
    tagged = [e for c in chunks for e in c.metadata.get("epics", [])]
    assert "SHS5213004" in tagged
    assert chunks[0].metadata["n_epics"] >= 1


def test_header_can_be_disabled():
    chunks = list(ElectoralRecordChunker(1, include_header=False).chunk(_doc()))
    # Repeating the header dilutes its IDF for BM25 (section-name queries fell
    # from 8.47 to 2.21), so opting out must stay possible.
    assert all(not c.text.startswith("विधानसभा निर्वाचन") for c in chunks[1:])


def test_page_without_records_still_yields_a_chunk():
    chunks = list(ElectoralRecordChunker().chunk(_doc(HEADER + "कोई रिकॉर्ड नहीं")))
    assert len(chunks) == 1


def test_empty_page_yields_nothing():
    assert list(ElectoralRecordChunker().chunk(_doc("   "))) == []


def test_rejects_bad_band_count():
    with pytest.raises(ValueError):
        ElectoralRecordChunker(bands_per_chunk=0)


def test_chunk_ids_are_unique():
    ids = [c.chunk_id for c in ElectoralRecordChunker(1).chunk(_doc())]
    assert len(ids) == len(set(ids))


# --- cascade ------------------------------------------------------------


def _sc(cid, score, rank):
    return ScoredChunk(cid, score, rank,
                       Chunk(chunk_id=cid, doc_id=cid, text="t", ordinal=0))


class _Index:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def search(self, query, limit=10):
        self.calls += 1
        return self.results[:limit]


LEX = [_sc("a", 9.0, 1), _sc("b", 4.0, 2)]
SEM = [_sc("b", 0.88, 1), _sc("c", 0.86, 2)]


def test_requires_a_bm25_index():
    with pytest.raises(ValueError):
        LexicalFirstRetriever(None)


def test_rejects_bad_candidate_limit():
    with pytest.raises(ValueError):
        LexicalFirstRetriever(_Index(LEX), candidate_limit=0)


def test_bm25_only_returns_lexical_hits():
    assert [h.chunk_id for h in LexicalFirstRetriever(_Index(LEX)).retrieve("q")] == ["a", "b"]


def test_lexical_silence_abstains_without_consulting_dense():
    """Dense answers everything, so consulting it here would end abstention."""
    dense = _Index(SEM)
    result = LexicalFirstRetriever(_Index([]), dense).retrieve("nonsense")
    assert result == []
    assert dense.calls == 0


def test_fusion_runs_only_when_bm25_found_something():
    dense = _Index(SEM)
    fused = LexicalFirstRetriever(_Index(LEX), dense).retrieve("q", limit=5)
    assert dense.calls == 1
    assert {h.chunk_id for h in fused} == {"a", "b", "c"}
    assert all(h.score < 1.0 for h in fused)  # RRF scale


def test_rescue_is_off_by_default():
    assert LexicalFirstRetriever(_Index([]), _Index(SEM)).retrieve("q") == []


def test_rescue_admits_only_hits_above_the_floor():
    dense = _Index([_sc("x", 0.84, 1), _sc("y", 0.79, 2)])
    hits = LexicalFirstRetriever(_Index([]), dense, dense_rescue=True,
                                 rescue_min_score=0.80).retrieve("q")
    assert [h.chunk_id for h in hits] == ["x"]


def test_rescue_still_abstains_when_everything_is_below_the_floor():
    """0.7923 was the highest nonsense score measured; the 0.80 floor excludes it."""
    dense = _Index([_sc("x", 0.7923, 1)])
    assert LexicalFirstRetriever(_Index([]), dense, dense_rescue=True).retrieve("q") == []


def test_default_rescue_floor_is_the_measured_value():
    assert LexicalFirstRetriever(_Index(LEX)).rescue_min_score == 0.80


def test_mode_names_the_active_pipeline():
    assert LexicalFirstRetriever(_Index(LEX)).mode == "bm25"
    assert LexicalFirstRetriever(_Index(LEX), _Index(SEM)).mode == "bm25+rrf"
    assert LexicalFirstRetriever(_Index(LEX), _Index(SEM),
                                 dense_rescue=True).mode == "bm25+rrf+rescue"


def test_non_positive_limit_returns_nothing():
    assert LexicalFirstRetriever(_Index(LEX)).retrieve("q", limit=0) == []
