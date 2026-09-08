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


def test_router_admin_metadata_classification():
    router = IntentRouter(lexical=None, dense=None)

    # English
    intent, _ = router.classify("What is the name of the polling booth and how many total voters?")
    assert intent == QueryIntent.ADMIN_METADATA

    intent, _ = router.classify("Tell me the booth name and total electors count")
    assert intent == QueryIntent.ADMIN_METADATA

    # Hindi
    intent, _ = router.classify("मतदान केंद्र का नाम क्या है और कुल मतदाता कितने हैं?")
    assert intent == QueryIntent.ADMIN_METADATA

    intent, _ = router.classify("इस भाग में मतदाताओं की कुल संख्या क्या है?")
    assert intent == QueryIntent.ADMIN_METADATA

    # Urdu
    intent, _ = router.classify("اس پولنگ بوتھ کا نام کیا ہے اور یہاں کل کتنے ووٹرز درج ہیں؟")
    assert intent == QueryIntent.ADMIN_METADATA

    intent, _ = router.classify("اس پولنگ اسٹیشن میں کل کتنے ووٹرز ہیں؟")
    assert intent == QueryIntent.ADMIN_METADATA


def test_router_admin_metadata_retrieval(tmp_path):
    db_path = tmp_path / "test_fts.db"
    fts = FTS5Index(db_path=db_path)
    p1 = Chunk(
        chunk_id="c_p1",
        doc_id="d1",
        text="मतदान स्थल का नाम: प्राथमिक विद्यालय रामपुर | मतदाताओं की कुल संख्या: 852",
        page=1,
        ordinal=0,
    )
    p20 = Chunk(
        chunk_id="c_p20",
        doc_id="d1",
        text="- [Serial: 12 | EPIC: SHS1234567 | Voter: अकरम]",
        page=20,
        ordinal=1,
    )
    fts.build([p1, p20])

    router = IntentRouter(lexical=fts, dense=None)
    results = router.retrieve("اس پولنگ بوتھ کا نام کیا ہے اور یہاں کل کتنے ووٹرز درج ہیں؟")
    assert len(results) >= 1
    assert results[0].chunk_id == "c_p1"
    assert "प्राथमिक विद्यालय रामपुर" in results[0].chunk.text


def test_router_spaced_and_hyphenated_ids():
    router = IntentRouter(lexical=None, dense=None)

    intent, params = router.classify("Find voter with ID JDK 6306765")
    assert intent == QueryIntent.EXACT_ENTITY
    assert params["entity_id"] == "JDK6306765"

    intent, params = router.classify("Voter card JDK-6306765 details")
    assert intent == QueryIntent.EXACT_ENTITY
    assert params["entity_id"] == "JDK6306765"


def test_router_serial_and_house_lookup(tmp_path):
    router = IntentRouter(lexical=None, dense=None)

    # Serial classification
    intent, params = router.classify("What are the details of serial 1088?")
    assert intent == QueryIntent.SERIAL_LOOKUP
    assert params["serial_num"] == "1088"

    intent, params = router.classify("क्रमांक 469 की जानकारी")
    assert intent == QueryIntent.SERIAL_LOOKUP
    assert params["serial_num"] == "469"

    # House classification
    intent, params = router.classify("Who lives in house number 4?")
    assert intent == QueryIntent.HOUSE_LOOKUP
    assert params["house_num"] == "4"

    intent, params = router.classify("मकान संख्या एस/0 के मतदाता")
    assert intent == QueryIntent.HOUSE_LOOKUP
    assert params["house_num"] == "एस/0"

    # Retrieval tests
    db_path = tmp_path / "test_fts.db"
    fts = FTS5Index(db_path=db_path)
    c1 = Chunk(chunk_id="c1", doc_id="d1", text="- [Serial: 1088 | EPIC: JDK6306765 | Voter: मो० अफरोज खान | House: एस/0]", ordinal=0)
    c2 = Chunk(chunk_id="c2", doc_id="d1", text="- [Serial: 469 | EPIC: JDK0926604 | Voter: अशफाक अहमद | House: 4]", ordinal=1)
    fts.build([c1, c2])

    router_with_fts = IntentRouter(lexical=fts, dense=None)
    hits = router_with_fts.retrieve("What are the details of serial 1088?")
    assert len(hits) >= 1
    assert "JDK6306765" in hits[0].chunk.text

    hits_house = router_with_fts.retrieve("Who lives in house number 4?")
    assert len(hits_house) >= 1
    assert "अशफाक अहमद" in hits_house[0].chunk.text


def test_router_bypasses_reranker_for_exact_intents(tmp_path):
    class MockReranker:
        def __init__(self):
            self.called = False

        def rerank(self, query, candidates, limit=None):
            self.called = True
            return candidates

    db_path = tmp_path / "test_fts.db"
    fts = FTS5Index(db_path=db_path)
    c1 = Chunk(chunk_id="c1", doc_id="d1", text="- [Serial: 1088 | EPIC: JDK6306765 | Voter: मो० अफरोज खान]", ordinal=0)
    fts.build([c1])

    mock_reranker = MockReranker()
    router = IntentRouter(lexical=fts, dense=None, reranker=mock_reranker)

    # EXACT_ENTITY query should bypass reranker
    results = router.retrieve("What are the details of voter ID JDK6306765?")
    assert len(results) >= 1
    assert mock_reranker.called is False  # Must not call cross-encoder for exact intent!
    assert results[0].score >= 0.95
