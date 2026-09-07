"""Lexical retrieval over chunks using rank_bm25 (Okapi BM25)."""

import re
import pickle
import unicodedata
from typing import Iterable, Optional

from rank_bm25 import BM25Okapi

from ingestion.documents import Chunk
from .types import ScoredChunk

# Tokenisation must survive non-Latin scripts.
#
# The original `[a-z0-9]+` discarded every Indic character, so the Hindi
# electoral rolls tokenised to nothing but the year digits and BM25 could never
# retrieve them. Plain `\w+` is not enough either: Python's `\w` follows
# str.isalnum(), which is False for combining marks (categories Mn/Mc), so
# "निर्वाचक" split at every matra into ['न','र','व','चक'].
#
# The class below is `\w` plus every combining mark in the range used by Indic
# scripts and Latin diacritics, so a word and its marks stay one token.
_MARK_RANGES = ((0x0300, 0x1B00), (0x1DC0, 0x1E00), (0x20D0, 0x20F1), (0xFE20, 0xFE30))
_MARKS = "".join(
    chr(cp)
    for lo, hi in _MARK_RANGES
    for cp in range(lo, hi)
    if unicodedata.category(chr(cp)) in ("Mn", "Mc", "Me")
)
_TOKEN = re.compile(rf"[\w{re.escape(_MARKS)}\/]+", re.UNICODE)

# Small closed-class list. BM25 already down-weights frequent terms via IDF, so
# this is mostly a speed/index-size win — keep it minimal to avoid dropping
# terms that matter in finance questions ("no", "own", "up", "down").
_ENGLISH_STOPWORDS = """
a an and are as at be by for from has have he in is it its of on that the to
was were will with this these those there their they i you your we our
"""

# Hindi function words. Without these the English-only stoplist let particles
# like "का" and "है" act as content terms: the nonsense queries
# "बिल्ली का बच्चा कहाँ सोता है" and "मेरी कार का इंजन खराब है" both scored
# 2.295 against the electoral rolls — above the legitimate query "कुम्हरार"
# (1.580). That destroys BM25's zero-result abstention signal, which is the
# cheapest and sharpest one this project has.
#
# Deliberately excludes domain content words (नाम, संख्या, क्षेत्र, मतदान,
# निर्वाचक), which carry the meaning in an electoral roll.
_HINDI_STOPWORDS = """
का के की को कि में से पर है हैं था थे थी हो होता होती होते ने और या भी ही तो
यह वह ये वे इस उस इन उन जो जब तब अगर तक साथ लिए बाद पहले नहीं कोई कुछ सब
अपना अपने अपनी मेरा मेरी मेरे तुम्हारा हमारा उनका उनकी इसका इसकी क्या कौन कहाँ
कब कैसे क्यों एक दो सकता सकती सकते करना करने किया गया गई गए रहा रही रहे
"""

STOPWORDS = frozenset(_ENGLISH_STOPWORDS.split()) | frozenset(_HINDI_STOPWORDS.split())


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode-aware tokenisation with stopword removal.

    Supports slashes inside identifiers (e.g. BR/35/207/291052) so that
    legacy electoral roll EPIC IDs remain single high-IDF tokens.
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        t = raw.strip("/")
        if t and t not in STOPWORDS:
            tokens.append(t)
            # If the token is a composite slash-separated ID, also index the terminal suffix
            # so queries for either the full ID or just the suffix number match.
            if "/" in t:
                suffix = t.split("/")[-1]
                if suffix and suffix != t and suffix not in STOPWORDS:
                    tokens.append(suffix)
    return tokens


class BM25Index:
    """In-memory BM25 index over a fixed chunk collection."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None

    def __len__(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Iterable[Chunk]) -> "BM25Index":
        self._chunks = list(chunks)
        if not self._chunks:
            raise ValueError("Cannot build a BM25 index over zero chunks")
        corpus = [tokenize(c.text) for c in self._chunks]
        # rank_bm25 divides by average document length; an all-empty corpus
        # would produce a ZeroDivisionError deep inside the library.
        if not any(corpus):
            raise ValueError("Every chunk tokenised to zero terms")
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, limit: int = 10) -> list[ScoredChunk]:
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[ScoredChunk] = []
        for rank, idx in enumerate(ranked[:limit], 1):
            # A zero score means no query term matched; returning it would pad
            # the candidate set with noise and distort fusion ranks.
            if scores[idx] <= 0:
                break
            results.append(
                ScoredChunk(
                    chunk_id=self._chunks[idx].chunk_id,
                    score=float(scores[idx]),
                    rank=rank,
                    chunk=self._chunks[idx],
                )
            )
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25,
                         "k1": self.k1, "b": self.b}, handle)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        index = cls(k1=state["k1"], b=state["b"])
        index._chunks = state["chunks"]
        index._bm25 = state["bm25"]
        return index
