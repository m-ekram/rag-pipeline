"""Offline tests for BM25, RRF fusion, hybrid wiring and Qdrant payload mapping.

The dense path is exercised against a fake Qdrant client so the whole suite
stays runnable without Docker or a downloaded embedding model.
"""

import pytest

from ingestion.documents import Chunk
from retrieval.bm25 import BM25Index, tokenize
from retrieval.dense import DenseIndex, point_id
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid import HybridRetriever
from retrieval.types import ScoredChunk


def _chunk(cid, text):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=text, ordinal=0)


CHUNKS = [
    _chunk("a::0", "Qdrant is an open source vector database for embeddings"),
    _chunk("b::0", "BM25 is a lexical ranking function used in search engines"),
    _chunk("c::0", "Reciprocal rank fusion combines several ranked result lists"),
    _chunk("d::0", "Mortgage interest is deductible on a primary residence"),
]


# --- BM25 ---------------------------------------------------------------


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Vector Database!") == ["vector", "database"]


def test_tokenize_keeps_finance_relevant_short_words():
    # These are load-bearing in finance questions and must survive the stoplist.
    for word in ("no", "own", "up", "down"):
        assert word in tokenize(f"should i {word} it")


def test_bm25_ranks_lexical_match_first():
    index = BM25Index().build(CHUNKS)
    hits = index.search("vector database", limit=3)
    assert hits[0].chunk_id == "a::0"
    assert hits[0].rank == 1


def test_bm25_excludes_zero_score_matches():
    index = BM25Index().build(CHUNKS)
    hits = index.search("mortgage", limit=10)
    # Only one chunk mentions mortgages; padding to limit=10 would be noise.
    assert [h.chunk_id for h in hits] == ["d::0"]


def test_bm25_returns_empty_for_all_stopword_query():
    index = BM25Index().build(CHUNKS)
    assert index.search("the and of", limit=5) == []


def test_bm25_requires_build_before_search():
    with pytest.raises(RuntimeError):
        BM25Index().search("anything")


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25Index().build([])


def test_bm25_roundtrips_through_disk(tmp_path):
    index = BM25Index().build(CHUNKS)
    path = str(tmp_path / "bm25.pkl")
    index.save(path)
    reloaded = BM25Index.load(path)
    assert len(reloaded) == len(CHUNKS)
    assert reloaded.search("vector database")[0].chunk_id == "a::0"


# --- RRF ----------------------------------------------------------------


def _sc(cid, rank):
    return ScoredChunk(chunk_id=cid, score=1.0 / rank, rank=rank)


def test_rrf_rewards_agreement_across_lists():
    # "b" tops neither list but is ranked 2nd in both; "a" tops the left list
    # and is buried at 5th in the right. Consensus wins on total rank.
    left = [_sc("a", 1), _sc("b", 2), _sc("c", 3)]
    right = [_sc("z", 1), _sc("b", 2), _sc("c", 3), _sc("y", 4), _sc("a", 5)]
    fused = reciprocal_rank_fusion([left, right])
    assert fused[0].chunk_id == "b"


def test_rrf_favours_rank_spread_over_consensus_at_equal_rank_sum():
    """Documents a real RRF gotcha rather than asserting the intuitive result.

    1/(k+r) is convex, so for an identical rank sum the *spread* scores higher:
    ranks 1+3 beat ranks 2+2. Consensus only wins when it lowers the rank sum.
    Worth knowing before tuning k or reading fusion results as "agreement".
    """
    # Fusion reads list position, so the padding is what sets the ranks.
    spread = [[_sc("a", 1)], [_sc("p", 1), _sc("q", 2), _sc("a", 3)]]
    consensus = [[_sc("p", 1), _sc("b", 2)], [_sc("q", 1), _sc("b", 2)]]
    a = next(f.score for f in reciprocal_rank_fusion(spread) if f.chunk_id == "a")
    b = next(f.score for f in reciprocal_rank_fusion(consensus) if f.chunk_id == "b")
    assert a > b
    assert a == pytest.approx(1 / 61 + 1 / 63)
    assert b == pytest.approx(2 / 62)


def test_rrf_scores_match_the_formula():
    fused = reciprocal_rank_fusion([[_sc("x", 1)], [_sc("x", 1)]], k=60)
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_uses_list_position_not_stale_rank_field():
    # A retriever that forgot to set .rank must still fuse correctly.
    left = [ScoredChunk("a", 9.0), ScoredChunk("b", 8.0)]
    fused = reciprocal_rank_fusion([left])
    assert [f.chunk_id for f in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_weights_shift_the_ordering():
    left = [_sc("a", 1), _sc("b", 2)]
    right = [_sc("b", 1), _sc("a", 2)]
    assert reciprocal_rank_fusion([left, right], weights=[3.0, 1.0])[0].chunk_id == "a"
    assert reciprocal_rank_fusion([left, right], weights=[1.0, 3.0])[0].chunk_id == "b"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[_sc("a", 1)]], weights=[1.0, 2.0])


def test_rrf_preserves_chunk_payload_from_whichever_list_has_it():
    bare = [ScoredChunk("a::0", 1.0)]
    rich = [ScoredChunk("a::0", 1.0, chunk=CHUNKS[0])]
    assert reciprocal_rank_fusion([bare, rich])[0].chunk is not None


def test_rrf_respects_limit_and_is_deterministic():
    left = [_sc(c, i) for i, c in enumerate("abcdef", 1)]
    fused = reciprocal_rank_fusion([left], limit=3)
    assert [f.chunk_id for f in fused] == ["a", "b", "c"]
    assert [f.rank for f in fused] == [1, 2, 3]


# --- dense / Qdrant mapping --------------------------------------------


def test_point_id_is_deterministic_and_uuid_shaped():
    first = point_id("3::0")
    assert first == point_id("3::0")
    assert first != point_id("3::1")
    assert len(first) == 36 and first.count("-") == 4


class _FakePoint:
    def __init__(self, payload, score):
        self.id, self.payload, self.score = payload["chunk_id"], payload, score


class _FakeResponse:
    def __init__(self, points):
        self.points = points


class _FakeClient:
    """Minimal stand-in for QdrantClient covering the calls DenseIndex makes."""

    def __init__(self):
        self.upserted, self.created = [], []

    def collection_exists(self, name):
        return False

    def create_collection(self, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config.size))

    def upsert(self, collection_name, points, wait=False):
        self.upserted.extend(points)

    def query_points(self, collection_name, query, limit, with_payload):
        return _FakeResponse([_FakePoint(p.payload, 0.9) for p in self.upserted[:limit]])


class _FakeEmbedder:
    dimension = 4

    def embed_passages(self, texts, show_progress=False):
        import numpy as np
        return np.ones((len(list(texts)), 4), dtype="float32")

    def embed_query(self, text):
        import numpy as np
        return np.ones(4, dtype="float32")


def test_dense_index_sizes_collection_from_the_model():
    client = _FakeClient()
    index = DenseIndex("c", _FakeEmbedder(), client=client)
    index.recreate()
    assert client.created == [("c", 4)]


def test_dense_index_batches_and_reports_count():
    client = _FakeClient()
    index = DenseIndex("c", _FakeEmbedder(), client=client)
    assert index.index(CHUNKS, batch_size=2) == 4
    assert len(client.upserted) == 4


def test_dense_search_rebuilds_chunks_from_payload():
    client = _FakeClient()
    index = DenseIndex("c", _FakeEmbedder(), client=client)
    index.index(CHUNKS)
    hits = index.search("vector database", limit=2)
    assert [h.chunk_id for h in hits] == ["a::0", "b::0"]
    assert hits[0].chunk.text == CHUNKS[0].text
    assert hits[0].rank == 1



class _FakeCrossEncoder:
    """Returns raw logits like ms-marco-MiniLM does."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, batch_size=32):
        return self.scores[: len(pairs)]


# --- hybrid wiring ------------------------------------------------------
#
# NOTE: HybridRetriever was refactored to a dense-only pipeline (commit d70c90b);
# BM25 and RRF are no longer fused into it. These tests were left asserting the
# old `HybridRetriever(bm25=..., dense=...)` signature and were failing at HEAD.
# BM25 survives as a standalone compared variant in the contamination sweep, so
# its tests above still stand.


def _dense():
    dense = DenseIndex("c", _FakeEmbedder(), client=_FakeClient())
    dense.index(CHUNKS)
    return dense


def test_hybrid_requires_a_dense_retriever():
    with pytest.raises(TypeError):
        HybridRetriever()
    with pytest.raises(ValueError):
        HybridRetriever(None)


def test_hybrid_rejects_a_non_positive_candidate_limit():
    with pytest.raises(ValueError):
        HybridRetriever(_dense(), candidate_limit=0)


def test_hybrid_mode_reports_active_pipeline():
    assert HybridRetriever(_dense()).mode == "dense"
    fake = _FakeCrossEncoder([1.0, 2.0, 3.0, 4.0])
    assert HybridRetriever(_dense(), reranker=fake).mode == "dense+rerank"


def test_hybrid_returns_dense_results_when_no_reranker():
    hits = HybridRetriever(_dense()).retrieve("vector database", limit=2)
    assert [h.chunk_id for h in hits] == ["a::0", "b::0"]


def test_hybrid_applies_the_reranker_and_reorders():
    from retrieval.reranker import CrossEncoderReranker as Reranker

    # Reverse the dense order via the cross-encoder's logits.
    reranker = Reranker(model=_FakeCrossEncoder([1.0, 2.0, 3.0, 4.0]))
    hits = HybridRetriever(_dense(), reranker=reranker).retrieve("q", limit=4)
    assert [h.chunk_id for h in hits] == ["d::0", "c::0", "b::0", "a::0"]
    assert all(0.0 < h.score < 1.0 for h in hits)


def test_hybrid_honours_a_non_positive_limit():
    assert HybridRetriever(_dense()).retrieve("q", limit=0) == []


def test_hybrid_retrieves_candidate_limit_deep_before_reranking():
    """The reranker needs a deeper candidate pool than the final result count."""
    from retrieval.reranker import CrossEncoderReranker as Reranker

    dense = _dense()
    retriever = HybridRetriever(dense, candidate_limit=4,
                                reranker=Reranker(model=_FakeCrossEncoder([1.0, 2.0, 3.0, 4.0])))
    assert len(retriever.retrieve("q", limit=2)) == 2
