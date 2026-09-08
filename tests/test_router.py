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


def test_router_exact_entity_prefix_fallback(tmp_path):
    db_path = tmp_path / "test_fts.db"
    fts = FTS5Index(db_path=db_path)
    c = Chunk(chunk_id="c1", doc_id="d1", text="- [Serial: 36 | EPIC: SHS5124394 | Voter: कशिश कश्यप]", ordinal=0)
    fts.build([c])

    router = IntentRouter(lexical=fts, dense=None)
    # Query with SHS5124391 (ending in 1 instead of 4 due to OCR variance)
    results = router.retrieve("What are the details of voter id SHS5124391?")
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert "कशिश कश्यप" in results[0].chunk.text


def test_router_exact_entity_fuzzy_levenshtein(tmp_path):
    db_path = tmp_path / "test_fts.db"
    fts = FTS5Index(db_path=db_path)
    # Stored EPIC has OCR confusion in middle digit: SHS4580493 vs query SHS4590493
    c = Chunk(chunk_id="c1", doc_id="d1", text="- [Serial: 10 | EPIC: SHS4580493 | Voter: फ़राज़ अहमद]", ordinal=0)
    fts.build([c])

    router = IntentRouter(lexical=fts, dense=None)
    # Query with SHS4590493 (distance 1 from stored SHS4580493)
    results = router.retrieve("What are the details of voter id SHS4590493?")
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert "फ़राज़ अहमद" in results[0].chunk.text

