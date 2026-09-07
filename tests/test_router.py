from ingestion.documents import Chunk
from retrieval.fts5_index import FTS5Index
from retrieval.router import IntentRouter, QueryIntent
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.types import ScoredChunk


def test_rrf_merging():
    c1 = Chunk(chunk_id="c1", doc_id="d1", text="text 1", ordinal=0)
    c2 = Chunk(chunk_id="c2", doc_id="d1", text="text 2", ordinal=1)
    c3 = Chunk(chunk_id="c3", doc_id="d1", text="text 3", ordinal=2)

    dense = [
        ScoredChunk(chunk_id="c1", score=0.9, rank=1, chunk=c1),
        ScoredChunk(chunk_id="c2", score=0.8, rank=2, chunk=c2),
    ]
    lexical = [
        ScoredChunk(chunk_id="c2", score=5.0, rank=1, chunk=c2),
        ScoredChunk(chunk_id="c3", score=4.0, rank=2, chunk=c3),
    ]

    fused = reciprocal_rank_fusion(dense, lexical, k=60, limit=3)
    assert len(fused) == 3
    # c2 was rank 2 in dense and rank 1 in lexical, so 1/62 + 1/61 = highest score
    assert fused[0].chunk_id == "c2"


def test_router_classification():
    router = IntentRouter(lexical=None, dense=None)

    intent, params = router.classify("What is on page 66?")
    assert intent == QueryIntent.PAGE_LOOKUP
    assert params["page_num"] == 66

    intent, params = router.classify("Who is the voter with ID BR/35/207/291052?")
    assert intent == QueryIntent.EXACT_ENTITY
    assert params["entity_id"] == "BR/35/207/291052"

    intent, params = router.classify("What are the land use guidelines for residential areas?")
    assert intent == QueryIntent.HYBRID_SEMANTIC
