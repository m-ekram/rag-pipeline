"""Query Understanding, Intent Routing, and Hybrid Retrieval Execution.

Routes queries based on intent:
- Physical Page Lookup ("What is on page 66?"): Direct payload filter bypassing embedding drift.
- Exact Entity / ID ("BR/35/207/291052", "SHS1291525"): Exact lexical matching with FTS5.
- Thematic / Semantic ("Proposed land use percentage in PPA"): Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import enum
import logging
import re
from typing import Optional, Protocol, Sequence

from ingestion.documents import Chunk
from .fts5_index import FTS5Index
from .rrf import reciprocal_rank_fusion
from .types import ScoredChunk

logger = logging.getLogger(__name__)

# Regex for detecting explicit page lookups: "page 66", "on page 12", "page no 3"
_PAGE_PATTERN = re.compile(r"(?:on\s+page|page\s+no\.?|page\s+number|\bp\.?)\s*(\d+)", re.IGNORECASE)

# Regex for detecting structured IDs: "BR/35/207/291052", "SHS5361415", "JDK 6306765", "JDK-6306765"
_ID_PATTERN = re.compile(
    r"\b([A-Z]{2,4}(?:\s*/\s*\d+){3}|[A-Z]{2,4}\s*[-/]?\s*[0-9OIl|BZS]{6,8})\b",
    re.IGNORECASE,
)

# Regex for detecting serial number queries: "serial 1088", "serial no: 469", "क्रमांक 1088", "سیریل نمبر 1088"
_SERIAL_PATTERN = re.compile(
    r"(?:(?:serial|sl\.?|voter\s*no\.?)\s*(?:no\.?|number)?|क्र(?:मांक|\.)?|سیریل\s*نمبر?)\s*[:#-]?\s*(\d{1,4})\b",
    re.IGNORECASE,
)

# Regex for detecting house number queries: "Who lives in house S/0?", "house number 4", "house no: S/0", "मकान संख्या 4", "مکان نمبر 4"
# `(?![a-z])` after each keyword stops "household" / "flatten" from reading as
# a house query whose "number" is the rest of the word.
_HOUSE_PATTERN = re.compile(
    r"(?:(?:who\s+lives\s+in|voters?\s+in|residents?\s+of)\s+house(?![a-z])\s*(?:no\.?|number)?|\b(?:house|h\.?\s*no\.?|quarter|flat)(?![a-z])\s*(?:no\.?|number)?|मकान\s*(?:संख्या|नं|नम्बर|सं\.)?|مکان\s*(?:نمبر)?)\s*[:#-]?\s*([०-९0-9A-Za-z\u0900-\u097F\/\-]+)",
    re.IGNORECASE,
)

# Regex for detecting relation queries: "father name Md Zahid Khan", "पिता का नाम मो० जाहिद खान", "husband name ...", etc.
_RELATION_PATTERN = re.compile(
    r"(?:(?:father|husband|mother|relative)(?:'s)?\s*(?:name)?|पिता(?:\s*का\s*नाम)?|पति(?:\s*का\s*नाम)?|माता(?:\s*का\s*नाम)?|संबंध|son\s+of|daughter\s+of|wife\s+of|\bs/o\b|\bd/o\b|\bw/o\b)\s*(?:is|was|named|name)?\s*[:#-]?\s*([A-Za-z\u0900-\u097F\s\.\u0964]+)",
    re.IGNORECASE,
)

# Regex for detecting exhaustive listing / aggregation queries: "Which voters...", "List all voters...", etc.
_EXHAUSTIVE_PATTERN = re.compile(
    r"(?:which\s+voters|list\s+(?:all\s+)?voters|all\s+voters|find\s+all\s+voters|who\s+all|सभी\s+मतदाता|किन\s+मतदाताओं|کل\s+ووٹرز)",
    re.IGNORECASE,
)

# Regex for detecting polling station / total voter count / administrative queries across English, Hindi, and Urdu
_ADMIN_PATTERN = re.compile(
    r"(?:polling\s*(?:booth|station)|booth\s*(?:name|number|address|location)|station\s*name|मतदान\s*(?:केंद्र|स्थल)|मतदाताओं\s*(?:की\s*)?कुल\s*संख्या|कुल\s*(?:मतदाता|वोटर)|total\s*(?:voters|electors)|number\s*of\s*voters|how\s*many\s*voters|voter\s*count|male\s*voters|female\s*voters|पुरुष\s*मतदाता|महिला\s*मतदाता|مرد\s*ووٹرز|خواتین\s*ووٹرز|پولنگ\s*(?:بوتھ|اسٹیشن|سٹیشن)|کل\s*(?:ووٹرز|رائے\s*دہندگان)|کتने\s*ووٹرز|ووٹنگ\s*لسٹ|انتخابی\s*فہرست|constituency|assembly|विधानसभा|भाग\s*संख्या|part\s*number|part\s*no|section\s*name|अनुभाग)",
    re.IGNORECASE,
)

_HINDI_TO_ENGLISH_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ENGLISH_TO_HINDI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")

_SCRIPT_SYNONYMS: dict[str, list[str]] = {
    "zahid": ["जाहिद", "ज़ाहिद", "जाहीद", "ज़ाहीद"],
    "khan": ["खान", "ख़ान", "कान"],
    "afroz": ["अफरोज", "अफ़रोज़", "अफ़रोज", "अफरोज़"],
    "faraz": ["फराज", "फ़राज़", "फेराक"],
    "tanveer": ["तनवीर"],
    "parvez": ["परवेज", "परवेज़"],
    "md": ["मो०", "मो", "मो.", "मो0", "मोहम्मद"],
    "mohammad": ["मोहम्मद", "मो०", "मो"],
    "ahmad": ["अहमद", "अहमद्"],
    "kumar": ["कुमार"],
    "kumari": ["कुमारी"],
    "devi": ["देवी"],
    "singh": ["सिंह"],
    "sharma": ["शर्मा"],
    "gupta": ["गुप्ता"],
    "kashyap": ["कश्यप"],
    "rai": ["राय"],
    "prasad": ["प्रसाद"],
    "ajay": ["अजय"],
    "amit": ["अमित"],
    "rahul": ["राहुल"],
    "kashish": ["कशिश"],
    "shreshtha": ["श्रेष्ठा"],
    "raj": ["राज"],
}

_REVERSE_SCRIPT_SYNONYMS: dict[str, list[str]] = {}
for en_word, hi_list in _SCRIPT_SYNONYMS.items():
    for hi_w in hi_list:
        _REVERSE_SCRIPT_SYNONYMS.setdefault(hi_w, []).append(en_word)


def expand_query_scripts(text: str) -> list[str]:
    """Expand English and Devanagari entities into dual-script search terms."""
    clean = text.strip()
    if not clean:
        return []

    terms = set()
    terms.add(clean)

    # House code expansions (e.g. S/0 -> एस/0, एस/ओ, एस/०)
    upper_t = clean.upper()
    if upper_t in ("S/0", "S/O", "S-0", "एस/0", "एस/ओ", "एस/०", "S 0"):
        terms.update(["S/0", "S/O", "एस/0", "एस/ओ", "एस/०", "S-0"])

    # Numeral digit swap (0-9 <-> ०-९)
    hi_dig = clean.translate(_ENGLISH_TO_HINDI_DIGITS)
    en_dig = clean.translate(_HINDI_TO_ENGLISH_DIGITS)
    terms.add(hi_dig)
    terms.add(en_dig)

    # Word-level transliteration / synonym expansion
    words = re.findall(r"[\w\u0900-\u097F]+", clean, re.UNICODE)
    for w in words:
        w_low = w.lower()
        if w_low in _SCRIPT_SYNONYMS:
            terms.update(_SCRIPT_SYNONYMS[w_low])
        if w in _REVERSE_SCRIPT_SYNONYMS:
            terms.update(_REVERSE_SCRIPT_SYNONYMS[w])

    return [t for t in terms if t]


def expand_query_groups(text: str) -> list[list[str]]:
    """Expand entity query into boolean AND groups of dual-script synonyms."""
    clean = text.strip()
    if not clean:
        return []

    # Check for house codes
    upper_t = clean.upper()
    if upper_t in ("S/0", "S/O", "S-0", "एस/0", "एस/ओ", "एस/०", "S 0"):
        return [["S/0", "S/O", "एस/0", "एस/ओ", "एस/०", "S-0"]]

    words = re.findall(r"[\w\u0900-\u097F]+", clean, re.UNICODE)
    stop_words = {"is", "was", "name", "whose", "who", "have", "has", "of", "ka", "ki", "ke", "का", "की", "के", "है", "हु"}
    content_words = [w for w in words if w.lower() not in stop_words]
    if not content_words:
        content_words = words

    substantive = [w for w in content_words if w.lower() not in ("md", "मो", "मो०", "मो.", "mr", "shri", "श्री")]
    target_words = substantive if substantive else content_words

    groups = []
    for w in target_words:
        w_low = w.lower()
        grp = set()
        grp.add(w)
        if w_low in _SCRIPT_SYNONYMS:
            grp.update(_SCRIPT_SYNONYMS[w_low])
        if w in _REVERSE_SCRIPT_SYNONYMS:
            grp.update(_REVERSE_SCRIPT_SYNONYMS[w])
        grp.add(w.translate(_ENGLISH_TO_HINDI_DIGITS))
        grp.add(w.translate(_HINDI_TO_ENGLISH_DIGITS))
        clean_grp = [t for t in grp if t and t != "मो"]
        if clean_grp:
            groups.append(clean_grp)

    return groups


class QueryIntent(enum.Enum):
    PAGE_LOOKUP = "page_lookup"
    EXACT_ENTITY = "exact_entity"
    SERIAL_LOOKUP = "serial_lookup"
    HOUSE_LOOKUP = "house_lookup"
    RELATION_LOOKUP = "relation_lookup"
    EXHAUSTIVE_LIST = "exhaustive_list"
    ADMIN_METADATA = "admin_metadata"
    HYBRID_SEMANTIC = "hybrid_semantic"


# A relation name ends where the next clause begins. The capture class admits
# spaces, so "father name Md Zahid Khan and live in house 4" captured "... and
# live in house", and every trailing word became a required search term.
_RELATION_TAIL = re.compile(
    r"\s+(?:and|or|who|which|whose|with|in|at|from|lives?|living|और|तथा|जो|में)(?=\s|$)[\s\S]*$",
    re.IGNORECASE,
)

# Intents that list every matching record rather than the single best one;
# their results must not be cut down to the caller's usual `limit`.
_ROSTER_INTENTS = frozenset({
    QueryIntent.EXHAUSTIVE_LIST, QueryIntent.RELATION_LOOKUP, QueryIntent.HOUSE_LOOKUP,
})
_ROSTER_LIMIT = 30


class IntentRouter:
    """Unified Query Intent Router and Hybrid Retriever."""

    def __init__(
        self,
        lexical: FTS5Index,
        dense,  # DenseIndex
        reranker=None,  # CrossEncoderReranker
        *,
        candidate_limit: int = 15,
    ):
        self.lexical = lexical
        self.dense = dense
        self.reranker = reranker
        self.candidate_limit = candidate_limit
        # Intent that served the most recent `retrieve` call (after any
        # fallback), so the pipeline can size evidence for list-style answers.
        self.last_intent: Optional[QueryIntent] = None

    def classify(self, query: str) -> tuple[QueryIntent, dict]:
        """Classify user intent and extract query parameters."""
        page_match = _PAGE_PATTERN.search(query)
        if page_match:
            return QueryIntent.PAGE_LOOKUP, {"page_num": int(page_match.group(1))}

        id_match = _ID_PATTERN.search(query)
        if id_match:
            clean_id = re.sub(r"[\s-]+", "", id_match.group(1)).upper()
            return QueryIntent.EXACT_ENTITY, {"entity_id": clean_id}

        serial_match = _SERIAL_PATTERN.search(query)
        if serial_match:
            return QueryIntent.SERIAL_LOOKUP, {"serial_num": serial_match.group(1)}

        rel_match = _RELATION_PATTERN.search(query)
        if rel_match:
            rel_name = _RELATION_TAIL.sub("", rel_match.group(1)).strip(" ?.,|:;\n")
            if len(rel_name) >= 2:
                return QueryIntent.RELATION_LOOKUP, {"relation_name": rel_name}

        house_match = _HOUSE_PATTERN.search(query)
        if house_match:
            return QueryIntent.HOUSE_LOOKUP, {"house_num": house_match.group(1)}

        if _ADMIN_PATTERN.search(query):
            return QueryIntent.ADMIN_METADATA, {}

        if _EXHAUSTIVE_PATTERN.search(query):
            return QueryIntent.EXHAUSTIVE_LIST, {}

        return QueryIntent.HYBRID_SEMANTIC, {}

    def retrieve(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        """Execute retrieval based on classified query intent."""
        intent, params = self.classify(query)

        candidates: list[ScoredChunk] = []

        if intent == QueryIntent.PAGE_LOOKUP:
            page_num = params["page_num"]
            logger.info("Routing query to PAGE_LOOKUP for page %d", page_num)
            page_chunks = []
            if self.lexical and hasattr(self.lexical, "get_by_page"):
                page_chunks = self.lexical.get_by_page(page_num)
            if not page_chunks and self.dense and hasattr(self.dense, "get_by_page"):
                page_chunks = self.dense.get_by_page(page_num)

            candidates = [
                ScoredChunk(chunk_id=c.chunk_id, score=1.0 - (idx * 0.01), rank=idx + 1, chunk=c)
                for idx, c in enumerate(page_chunks[: self.candidate_limit])
            ]

        elif intent == QueryIntent.EXACT_ENTITY:
            entity_id = params["entity_id"]
            logger.info("Routing query to EXACT_ENTITY for ID '%s'", entity_id)
            lexical_hits = self.lexical.search(entity_id, limit=self.candidate_limit) if self.lexical else []
            if not lexical_hits and len(entity_id) >= 6 and self.lexical:
                prefix_query = entity_id[:-1] + "*"
                logger.info("Exact ID miss. Trying prefix search: '%s'", prefix_query)
                lexical_hits = self.lexical.search(prefix_query, limit=self.candidate_limit)
            if not lexical_hits and self.lexical and hasattr(self.lexical, "fuzzy_search_epic"):
                logger.info("Exact and prefix miss. Trying Levenshtein fuzzy search for: '%s'", entity_id)
                lexical_hits = self.lexical.fuzzy_search_epic(entity_id, max_distance=2, limit=self.candidate_limit)
            if lexical_hits:
                candidates = [
                    ScoredChunk(chunk_id=c.chunk_id, score=max(1.0 - (idx * 0.01), c.score if c.score <= 1.0 else 1.0), rank=idx + 1, chunk=c.chunk)
                    for idx, c in enumerate(lexical_hits)
                ]
            else:
                # If exact ID didn't hit in lexical, fall back to hybrid search
                intent = QueryIntent.HYBRID_SEMANTIC

        elif intent == QueryIntent.SERIAL_LOOKUP:
            serial_num = params["serial_num"]
            logger.info("Routing query to SERIAL_LOOKUP for Serial '%s'", serial_num)
            lexical_hits = []
            if self.lexical:
                lexical_hits = self.lexical.search(f'"Serial: {serial_num}"', limit=self.candidate_limit)
                if not lexical_hits:
                    lexical_hits = self.lexical.search(f"s{serial_num}", limit=self.candidate_limit)
            if lexical_hits:
                candidates = [
                    ScoredChunk(chunk_id=c.chunk_id, score=1.0 - (idx * 0.01), rank=idx + 1, chunk=c.chunk)
                    for idx, c in enumerate(lexical_hits)
                ]
            else:
                intent = QueryIntent.HYBRID_SEMANTIC

        elif intent == QueryIntent.RELATION_LOOKUP:
            rel_name = params["relation_name"]
            logger.info("Routing query to RELATION_LOOKUP for Relation '%s'", rel_name)
            expanded_groups = expand_query_groups(rel_name)
            lexical_hits = []
            if self.lexical and hasattr(self.lexical, "search_field"):
                lexical_hits = self.lexical.search_field("Relation", expanded_groups, limit=max(limit, 30))
            if not lexical_hits and self.lexical:
                lexical_hits = self.lexical.search(f'"{rel_name}"', limit=max(limit, 30))
            if lexical_hits:
                candidates = [
                    ScoredChunk(chunk_id=c.chunk_id, score=max(1.0 - (idx * 0.005), c.score if c.score <= 1.0 else 1.0), rank=idx + 1, chunk=c.chunk)
                    for idx, c in enumerate(lexical_hits)
                ]
            else:
                intent = QueryIntent.HYBRID_SEMANTIC

        elif intent == QueryIntent.HOUSE_LOOKUP:
            house_num = params["house_num"]
            logger.info("Routing query to HOUSE_LOOKUP for House '%s'", house_num)
            expanded_groups = expand_query_groups(house_num)
            lexical_hits = []
            if self.lexical and hasattr(self.lexical, "search_field"):
                lexical_hits = self.lexical.search_field("House", expanded_groups, limit=max(limit, 30))
            if not lexical_hits and self.lexical:
                for grp in expanded_groups:
                    for term in grp:
                        lexical_hits = self.lexical.search(f'"{term}"', limit=max(limit, 30))
                        if lexical_hits:
                            break
                    if lexical_hits:
                        break
            if lexical_hits:
                candidates = [
                    ScoredChunk(chunk_id=c.chunk_id, score=max(1.0 - (idx * 0.005), c.score if c.score <= 1.0 else 1.0), rank=idx + 1, chunk=c.chunk)
                    for idx, c in enumerate(lexical_hits)
                ]
            else:
                intent = QueryIntent.HYBRID_SEMANTIC

        elif intent == QueryIntent.ADMIN_METADATA:
            logger.info("Routing query to ADMIN_METADATA (Page 1 polling station & elector summary)")
            page1_chunks = []
            if self.lexical and hasattr(self.lexical, "get_by_page"):
                page1_chunks = self.lexical.get_by_page(1)
            if not page1_chunks and self.dense and hasattr(self.dense, "get_by_page"):
                page1_chunks = self.dense.get_by_page(1)
            summary_hits = []
            if self.lexical and hasattr(self.lexical, "search"):
                summary_hits = self.lexical.search("मतदाताओं की कुल संख्या", limit=3)
            seen_ids = set()
            all_chunks = []
            for c in page1_chunks:
                seen_ids.add(c.chunk_id)
                all_chunks.append(c)
            for sc in summary_hits:
                if sc.chunk and sc.chunk.chunk_id not in seen_ids:
                    seen_ids.add(sc.chunk.chunk_id)
                    all_chunks.append(sc.chunk)
            if all_chunks:
                candidates = [
                    ScoredChunk(chunk_id=c.chunk_id, score=1.0 - (idx * 0.01), rank=idx + 1, chunk=c)
                    for idx, c in enumerate(all_chunks[: self.candidate_limit])
                ]
            else:
                intent = QueryIntent.HYBRID_SEMANTIC

        # List-style intents keep every match they found (up to the roster cap);
        # slicing a relation or house lookup to `limit` silently dropped voters.
        effective_limit = max(limit, _ROSTER_LIMIT) if intent in _ROSTER_INTENTS else limit

        if intent in (QueryIntent.HYBRID_SEMANTIC, QueryIntent.EXHAUSTIVE_LIST) or not candidates:
            logger.info("Routing query to HYBRID_SEMANTIC / EXHAUSTIVE_LIST (RRF k=60, limit=%d)", effective_limit)
            dense_hits = self.dense.search(query, limit=max(self.candidate_limit, effective_limit)) if self.dense else []
            lexical_hits = self.lexical.search(query, limit=max(self.candidate_limit, effective_limit)) if self.lexical else []

            candidates = reciprocal_rank_fusion(
                dense_hits, lexical_hits, k=60, limit=effective_limit
            )

        self.last_intent = intent

        # Cross-encoder neural reranking:
        # Crucial architectural guard: ONLY rerank fuzzy semantic queries (HYBRID_SEMANTIC).
        # Deterministic structural/exact lookups (EXACT_ENTITY, SERIAL_LOOKUP, HOUSE_LOOKUP,
        # RELATION_LOOKUP, PAGE_LOOKUP, ADMIN_METADATA) have exact factual relevance (score >= 1.0)
        # and must NEVER be penalized or scrambled by a conversational sentence-similarity model.
        if intent == QueryIntent.HYBRID_SEMANTIC and self.reranker and candidates:
            has_exact = any(c.score >= 0.95 for c in candidates)
            if not has_exact:
                reranked = self.reranker.rerank(query, candidates, limit=limit)
                return reranked

        return candidates[:effective_limit]
