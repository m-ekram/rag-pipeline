"""Tests for the retrieval metrics.

These were previously untested, which is a problem for a project whose
definition of done is "no number is asserted without a script that produced
it" — an incorrect metric silently invalidates every reported result.
"""

import math

import pytest

from eval.retrieval_eval import (
    _document_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ingestion.documents import Chunk
from retrieval.types import ScoredChunk

RANKING = ["d1", "x", "d2", "y", "z"]      # relevant at ranks 1 and 3
RELEVANCE = {"d1": 1, "d2": 1, "d3": 1}    # d3 is never retrieved
RELEVANT = {"d1", "d2", "d3"}


# --- precision ----------------------------------------------------------


def test_precision_at_k():
    assert precision_at_k(RANKING, RELEVANT, 1) == pytest.approx(1.0)
    assert precision_at_k(RANKING, RELEVANT, 2) == pytest.approx(0.5)
    assert precision_at_k(RANKING, RELEVANT, 5) == pytest.approx(0.4)


def test_precision_divides_by_k_not_by_returned_count():
    """A short ranking must not score as perfect precision."""
    assert precision_at_k(["d1"], RELEVANT, 10) == pytest.approx(0.1)


def test_precision_of_empty_ranking_is_zero():
    assert precision_at_k([], RELEVANT, 10) == 0.0
    assert precision_at_k(RANKING, RELEVANT, 0) == 0.0


# --- recall -------------------------------------------------------------


def test_recall_at_k():
    assert recall_at_k(RANKING, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKING, RELEVANT, 1) == pytest.approx(1 / 3)


def test_recall_with_no_relevant_documents_is_zero():
    assert recall_at_k(RANKING, set(), 5) == 0.0


def test_recall_is_capped_at_one():
    assert recall_at_k(["d1", "d1", "d2", "d3"], RELEVANT, 10) == pytest.approx(1.0)


# --- MRR ----------------------------------------------------------------


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(RANKING, RELEVANT, 5) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "d1"], RELEVANT, 5) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y", "d2"], RELEVANT, 5) == pytest.approx(1 / 3)


def test_reciprocal_rank_respects_the_cutoff():
    assert reciprocal_rank(["x", "y", "d1"], RELEVANT, 2) == 0.0


def test_reciprocal_rank_with_no_hit_is_zero():
    assert reciprocal_rank(["x", "y"], RELEVANT, 5) == 0.0


# --- nDCG ---------------------------------------------------------------


def test_ndcg_matches_hand_computed_value():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKING, RELEVANCE, 5) == pytest.approx(dcg / idcg)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["d1", "d2", "d3"], RELEVANCE, 3) == pytest.approx(1.0)
    assert ndcg_at_k(RANKING, RELEVANCE, 1) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k(["x", "y"], RELEVANCE, 5) == 0.0


def test_ndcg_with_empty_relevance_is_zero():
    assert ndcg_at_k(RANKING, {}, 5) == 0.0


def test_ndcg_ideal_is_truncated_at_k():
    """IDCG@1 must consider only the single best gain, or nDCG can never reach 1."""
    assert ndcg_at_k(["d1"], RELEVANCE, 1) == pytest.approx(1.0)


# --- chunk -> document collapsing ---------------------------------------


def _sc(chunk_id, doc_id, score):
    return ScoredChunk(chunk_id, score, 0,
                       Chunk(chunk_id=chunk_id, doc_id=doc_id, text="t", ordinal=0))


def test_document_ranking_collapses_chunks_keeping_best_rank():
    results = [_sc("d1::0", "d1", 0.9), _sc("d1::1", "d1", 0.8), _sc("d2::0", "d2", 0.7)]
    assert _document_ranking(results) == ["d1", "d2"]


def test_document_ranking_skips_results_without_payload():
    results = [ScoredChunk("x::0", 0.9), _sc("d1::0", "d1", 0.8)]
    assert _document_ranking(results) == ["d1"]


def test_document_ranking_of_nothing_is_empty():
    assert _document_ranking([]) == []
