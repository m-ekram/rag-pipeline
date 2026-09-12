"""Disk-backed SQLite FTS5 lexical index for large-scale RAG (17k+ pages).

Supports:
- Punctuation & slash-safe tokenization (preserves legacy EPIC IDs like BR/35/207/291052)
- Native BM25 ranking via SQLite FTS5 `bm25()`
- Payload metadata storage and optional doc_id/page_num filtering
- In-memory or on-disk persistence for zero RAM bloat at scale
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ingestion.documents import Chunk
from .types import ScoredChunk

logger = logging.getLogger(__name__)

# Token pattern: matches letters, digits, and characters like / - _ . *
_TOKEN_PATTERN = re.compile(r"[\ẁ-ᬀ/_\-.*]+", re.UNICODE)

_CHUNK_FIELDS = frozenset(Chunk.__dataclass_fields__)


def _quote(term: str) -> str:
    """Render one term as an FTS5 string literal.

    Quoted, a term can never be read as query syntax. Bare AND / OR / NOT /
    NEAR are operators and a bare slash is a syntax error; either raised an
    OperationalError that surfaced as zero results.
    """
    return '"' + term.replace('"', '""') + '"'


def _row_to_chunk(cid: str, text: str, meta_json: Optional[str]) -> Chunk:
    """Rebuild a Chunk from a stored row.

    The payload JSON holds every field except the text, which lives only in the
    indexed `text` column.
    """
    payload = json.loads(meta_json) if meta_json else {}
    kwargs = {k: v for k, v in payload.items() if k in _CHUNK_FIELDS}
    kwargs.setdefault("chunk_id", cid)
    kwargs.setdefault("doc_id", cid.split("::")[0] if "::" in cid else cid)
    kwargs["text"] = text
    kwargs.setdefault("ordinal", 0)
    return Chunk(**kwargs)


class FTS5Index:
    """SQLite FTS5 lexical index over chunks."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize virtual FTS5 table with unicode61 tokenizer preserving slashes."""
        with self.con:
            self.con.execute("DROP TABLE IF EXISTS chunks_fts")
            self.con.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    page_num UNINDEXED,
                    text,
                    metadata_json UNINDEXED,
                    tokenize="unicode61 tokenchars '/_-.'"
                )
                """
            )

    def build(self, chunks: Iterable[Chunk]) -> FTS5Index:
        """Populate the index with chunks."""
        self._init_db()
        items = list(chunks)
        if not items:
            return self

        rows = []
        for c in items:
            meta = c.metadata or {}
            page_val = getattr(c, "page", None) or meta.get("page") or meta.get("page_num", 0)
            try:
                page_num = int(page_val) if page_val is not None else 0
            except (ValueError, TypeError):
                page_num = 0
            payload = c.to_payload()
            # The text is stored once, in the indexed column; a second copy in
            # the payload doubled every row. Unescaped UTF-8 keeps Devanagari at
            # 3 bytes a character instead of 6.
            payload.pop("text", None)
            rows.append(
                (
                    c.chunk_id,
                    c.doc_id,
                    page_num,
                    c.text,
                    json.dumps(payload, ensure_ascii=False),
                )
            )

        with self.con:
            self.con.executemany(
                """
                INSERT INTO chunks_fts (chunk_id, doc_id, page_num, text, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        logger.info("FTS5Index built with %d chunks (db: %s)", len(rows), self.db_path)
        return self

    def _format_query(self, query: str) -> str:
        """Convert natural language query into FTS5-safe syntax."""
        # Extract meaningful tokens (words, alphanumeric IDs with slashes)
        tokens = _TOKEN_PATTERN.findall(query)
        if not tokens:
            return ""

        formatted = []
        for t in tokens:
            cleaned = t.strip("./-_*")
            if not cleaned:
                continue
            # A '*' after a quoted string is FTS5's prefix query, so both
            # "SHS512439"* and "BR/35/207/29105"* work as prefixes.
            if t.endswith("*"):
                formatted.append(_quote(t.rstrip("*")) + "*")
            else:
                formatted.append(_quote(t))

        if not formatted:
            return ""

        # Use OR logic between terms so partial matches still retrieve candidates,
        # with FTS5 BM25 naturally ranking exact matches highest.
        return " OR ".join(formatted)

    def search(
        self,
        query: str,
        limit: int = 15,
        doc_id: Optional[str] = None,
        page_num: Optional[int] = None,
    ) -> list[ScoredChunk]:
        """Search the FTS5 index and return ScoredChunk items ranked by BM25."""
        fts_query = self._format_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT chunk_id, text, metadata_json, bm25(chunks_fts) as rank_score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
        """
        params: list[object] = [fts_query]

        if doc_id is not None:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        if page_num is not None:
            sql += " AND page_num = ?"
            params.append(page_num)

        sql += " ORDER BY rank_score ASC LIMIT ?"
        params.append(limit)

        try:
            cursor = self.con.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query '%s' failed: %s", fts_query, e)
            return []

        # SQLite FTS5 bm25() is negative with lower = better (e.g. -15.2 beats
        # -3.1); negate it so a higher score is better.
        return [
            ScoredChunk(chunk_id=cid, score=-float(raw_score), rank=rank,
                        chunk=_row_to_chunk(cid, text, meta_json))
            for rank, (cid, text, meta_json, raw_score) in enumerate(rows, 1)
        ]

    def search_field(
        self,
        field_name: str,
        terms: Sequence[str],
        limit: int = 50,
        doc_id: Optional[str] = None,
    ) -> list[ScoredChunk]:
        """Exhaustively search chunks for records matching specific fields and expanded terms.

        Uses FTS5 Boolean expressions and fallback prefix / SQL LIKE matching to ensure zero missed records.
        """
        if not terms:
            return []

        field_markers = []
        fn_lower = field_name.lower()
        if fn_lower in ("relation", "father", "husband", "mother", "rel"):
            field_markers = ["Relation", "पिता", "पति", "माता", "संबंध"]
        elif fn_lower in ("house", "makan", "ghar"):
            field_markers = ["House", "मकान", "घर"]
        elif fn_lower in ("voter", "name", "naam"):
            field_markers = ["Voter", "नाम"]
        elif fn_lower in ("serial", "kramank"):
            field_markers = ["Serial", "क्रमांक", "क्रम"]
        else:
            field_markers = [field_name]

        # Terms are either flat alternatives, or groups of dual-script variants
        # (e.g. [["Zahid", "जाहिद"], ["Khan", "खान"]]) of which every group is
        # required.
        grouped = isinstance(terms[0], (list, tuple, set))
        raw_groups = terms if grouped else [terms]
        groups = [[str(t).strip() for t in grp if str(t).strip()] for grp in raw_groups]
        groups = [grp for grp in groups if grp]
        if not groups:
            return []

        fm_clause = " OR ".join(_quote(m) for m in field_markers)
        terms_expr = " AND ".join(
            "(" + " OR ".join(_quote(t) for t in grp) + ")" for grp in groups
        )
        queries_to_try = [f"({fm_clause}) AND {terms_expr}", terms_expr]

        seen_ids: set[str] = set()
        results: list[ScoredChunk] = []

        def add(cid: str, text: str, meta_json: Optional[str]) -> bool:
            """Record a hit once; True when the limit is reached."""
            if cid not in seen_ids:
                seen_ids.add(cid)
                results.append(ScoredChunk(
                    chunk_id=cid,
                    score=1.0 - (len(results) * 0.005),
                    rank=len(results) + 1,
                    chunk=_row_to_chunk(cid, text, meta_json),
                ))
            return len(results) >= limit

        for fts_q in queries_to_try:
            sql = """
                SELECT chunk_id, text, metadata_json, bm25(chunks_fts) as rank_score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
            """
            params: list[object] = [fts_q]
            if doc_id is not None:
                sql += " AND doc_id = ?"
                params.append(doc_id)
            sql += " ORDER BY rank_score ASC LIMIT ?"
            params.append(limit)

            try:
                rows = self.con.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                logger.debug("FTS5 search_field query '%s' error: %s", fts_q, e)
                continue
            for cid, text, meta_json, _raw_score in rows:
                if add(cid, text, meta_json):
                    return results

        # Substring fallback for words FTS cannot see as tokens: OCR often glues
        # a name to its neighbours. Every group is still required, so "Khan"
        # alone does not match a search for "Zahid Khan".
        if not results:
            clauses: list[str] = []
            like_params: list[object] = []
            for grp in groups:
                variants = [v for v in (t.strip("\"'") for t in grp) if len(v) >= 2]
                if variants:
                    clauses.append("(" + " OR ".join("text LIKE ?" for _ in variants) + ")")
                    like_params.extend(f"%{v}%" for v in variants)
            if clauses:
                sql = ("SELECT chunk_id, text, metadata_json FROM chunks_fts WHERE "
                       + " AND ".join(clauses))
                if doc_id is not None:
                    sql += " AND doc_id = ?"
                    like_params.append(doc_id)
                sql += " LIMIT ?"
                like_params.append(limit)
                for cid, text, meta_json in self.con.execute(sql, like_params).fetchall():
                    if add(cid, text, meta_json):
                        break

        return results

    def get_by_page(self, page_num: int, doc_id: Optional[str] = None) -> list[Chunk]:
        """Fetch all chunks belonging to a specific page number."""
        sql = "SELECT chunk_id, text, metadata_json FROM chunks_fts WHERE page_num = ?"
        params: list[object] = [page_num]
        if doc_id is not None:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY rowid ASC"

        cursor = self.con.execute(sql, params)
        return [_row_to_chunk(cid, text, meta_json) for cid, text, meta_json in cursor.fetchall()]

    def fuzzy_search_epic(
        self, target_epic: str, max_distance: int = 2, limit: int = 5
    ) -> list[ScoredChunk]:
        """Fuzzy search across voter chunks for EPIC IDs with optical normalization and Levenshtein distance."""
        clean_target = _normalize_optical_epic(target_epic)
        if len(clean_target) < 6:
            return []

        # Extract numeric core if present (e.g. '6306765' from 'JDK6306765')
        num_core = re.search(r"\d{5,8}", clean_target)
        target_digits = num_core.group(0) if num_core else ""

        if target_digits and len(target_digits) >= 5:
            cursor = self.con.execute(
                "SELECT chunk_id, text, metadata_json FROM chunks_fts WHERE text LIKE '%EPIC:%' OR text LIKE ?",
                (f"%{target_digits}%",),
            )
        else:
            cursor = self.con.execute(
                "SELECT chunk_id, text, metadata_json FROM chunks_fts WHERE text LIKE '%EPIC:%'"
            )
        rows = cursor.fetchall()

        matches = []
        for cid, text, meta_json in rows:
            epics = re.findall(r"EPIC:\s*([^\s|,]+)", text)
            best_dist = 999
            matched_digit = False

            if target_digits and len(target_digits) >= 5 and target_digits in text:
                matched_digit = True
                best_dist = 0
            else:
                for ep in epics:
                    clean_ep = _normalize_optical_epic(ep)
                    if target_digits and target_digits in clean_ep:
                        matched_digit = True
                        best_dist = 0
                        break

                    if abs(len(clean_ep) - len(clean_target)) > max_distance:
                        continue
                    dist = _levenshtein(clean_target, clean_ep)
                    if dist < best_dist:
                        best_dist = dist

            if matched_digit or best_dist <= max_distance:
                chunk = _row_to_chunk(cid, text, meta_json)
                score = 1.0 if matched_digit else 1.0 - (best_dist * 0.05)
                matches.append((best_dist, ScoredChunk(chunk_id=cid, score=score, rank=1, chunk=chunk)))

        matches.sort(key=lambda x: x[0])
        return [m[1] for m in matches[:limit]]

    def close(self) -> None:
        """Close SQLite connection."""
        self.con.close()


def _normalize_optical_epic(raw: str) -> str:
    """Normalize optical character confusions in voter EPIC numbers."""
    clean = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    # Normalize common OCR misreads at start: e.g. J0K -> JDK, 1DK -> JDK, 5HS -> SHS, 20 -> JDK
    clean = re.sub(r"^(?:J0K|1DK|UDK|4JDK|20|2O|2D)", "JDK", clean)
    clean = re.sub(r"^(?:5HS|\$HS)", "SHS", clean)

    m = re.match(r"^([A-Z]{2,4})(.*)$", clean)
    if m:
        pref, rest = m.groups()
        rest_norm = (
            rest.replace("I", "1")
            .replace("L", "1")
            .replace("O", "0")
            .replace("D", "0")
            .replace("S", "5")
            .replace("B", "8")
            .replace("Z", "2")
        )
        return pref + rest_norm
    return clean


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
