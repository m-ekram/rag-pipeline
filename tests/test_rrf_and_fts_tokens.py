"""Regression tests for bugs found in the Swiss-Army-Knife rewrite.

Each of these was silent: no exception, no warning, just degraded retrieval.
"""

import re

import pytest

from ingestion.documents import Chunk
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.types import ScoredChunk
from storage.fts import _TOKEN_PATTERN as STORAGE_TOKENS
from retrieval.fts5_index import _TOKEN_PATTERN as FTS5_TOKENS

HINDI = "निर्वाचक नामावली मतदान केंद्र कुम्हरार"


def _c(cid):
    return Chunk(chunk_id=cid, doc_id=cid, text=f"text-{cid}", ordinal=0)


# --- RRF ----------------------------------------------------------------


def test_rrf_uses_list_position_not_the_rank_field():
    """ScoredChunk.rank defaults to 0, so trusting it gave every result
    1/(k+0) — identical scores, ranking collapsed to insertion order."""
    dense = [ScoredChunk("a", 0.9), ScoredChunk("b", 0.8), ScoredChunk("c", 0.7)]
    lexical = [ScoredChunk("c", 5.0), ScoredChunk("b", 4.0), ScoredChunk("a", 3.0)]

    fused = reciprocal_rank_fusion(dense, lexical)
    assert len({round(f.score, 9) for f in fused}) > 1, "scores must not all tie"
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_preserves_the_payload_from_whichever_list_has_it():
    """A dense hit with a missing Qdrant payload used to overwrite a lexical
    hit that had the text, and the evidence vanished downstream."""
    dense = [ScoredChunk("x", 0.9, 1, None)]
    lexical = [ScoredChunk("x", 5.0, 1, _c("x"))]
    assert reciprocal_rank_fusion(dense, lexical)[0].chunk is not None


def test_rrf_rewards_agreement():
    dense = [ScoredChunk("a", 0.9, 1), ScoredChunk("b", 0.8, 2)]
    lexical = [ScoredChunk("b", 5.0, 1), ScoredChunk("a", 4.0, 2)]
    fused = reciprocal_rank_fusion(dense, lexical)
    assert {f.chunk_id for f in fused} == {"a", "b"}
    assert all(f.score == pytest.approx(1 / 61 + 1 / 62) for f in fused)


def test_rrf_is_deterministic_on_ties():
    dense = [ScoredChunk("b", 0.9, 1), ScoredChunk("a", 0.8, 2)]
    lexical = [ScoredChunk("a", 5.0, 1), ScoredChunk("b", 4.0, 2)]
    first = [f.chunk_id for f in reciprocal_rank_fusion(dense, lexical)]
    assert first == [f.chunk_id for f in reciprocal_rank_fusion(dense, lexical)]


def test_rrf_renumbers_ranks_from_one():
    dense = [ScoredChunk("a", 0.9, 1), ScoredChunk("b", 0.8, 2)]
    assert [f.rank for f in reciprocal_rank_fusion(dense, [])] == [1, 2]


def test_rrf_respects_limit():
    dense = [ScoredChunk(c, 1.0, i) for i, c in enumerate("abcdef", 1)]
    assert len(reciprocal_rank_fusion(dense, [], limit=3)) == 3


def test_rrf_handles_one_empty_side():
    dense = [ScoredChunk("a", 0.9, 1)]
    assert [f.chunk_id for f in reciprocal_rank_fusion(dense, [])] == ["a"]
    assert [f.chunk_id for f in reciprocal_rank_fusion([], dense)] == ["a"]
    assert reciprocal_rank_fusion([], []) == []


# --- FTS tokenisation ---------------------------------------------------


def test_storage_fts_keeps_devanagari_words_whole():
    """`[\\w/.-]+` shattered "निर्वाचक" into ['न','र','व','चक'], so Hindi
    search through SQLiteFTS matched nothing."""
    assert STORAGE_TOKENS.findall(HINDI) == [
        "निर्वाचक", "नामावली", "मतदान", "केंद्र", "कुम्हरार",
    ]


def test_both_fts_tokenisers_agree_on_devanagari():
    """Two implementations existed; only one had the fix."""
    assert STORAGE_TOKENS.findall(HINDI) == FTS5_TOKENS.findall(HINDI)


def test_storage_fts_still_handles_ids_and_english():
    assert STORAGE_TOKENS.findall("BR/35/207/291052") == ["BR/35/207/291052"]
    assert STORAGE_TOKENS.findall("SHS5361415") == ["SHS5361415"]
    assert STORAGE_TOKENS.findall("node level facilities") == [
        "node", "level", "facilities",
    ]


# --- script import safety ----------------------------------------------


@pytest.mark.parametrize("module", ["index_booths", "scripts.inspect_fts"])
def test_scripts_do_not_execute_on_import(module, capsys):
    """Both printed output and did real work at import time."""
    import importlib

    importlib.import_module(module)
    assert capsys.readouterr().out == ""
