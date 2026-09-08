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
_TOKEN_PATTERN = re.compile(r"[\w\u0300-\u1B00/_\-.*]+", re.UNICODE)


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
            rows.append(
                (
                    c.chunk_id,
                    c.doc_id,
                    page_num,
                    c.text,
                    json.dumps(c.to_payload()),
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

        # Quote tokens containing slashes, dashes, or special chars
        formatted = []
        for t in tokens:
            cleaned = t.strip("./-_*")
            if not cleaned:
                continue
            if any(ch in t for ch in "/-_.") and not t.endswith("*"):
                formatted.append(f'"{t}"')
            else:
                formatted.append(t)

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

        scored: list[ScoredChunk] = []
        for rank, (cid, text, meta_json, raw_score) in enumerate(rows, 1):
            # SQLite FTS5 bm25() returns negative values where lower is better (e.g. -15.2 is better than -3.1).
            # Convert to positive score: -raw_score so higher is better.
            score = -float(raw_score)
            payload = json.loads(meta_json) if meta_json else {}
            chunk_fields = {f for f in Chunk.__dataclass_fields__}
            chunk_kwargs = {k: v for k, v in payload.items() if k in chunk_fields}
            chunk_kwargs.setdefault("chunk_id", cid)
            chunk_kwargs.setdefault("doc_id", cid.split("::")[0] if "::" in cid else cid)
            chunk_kwargs.setdefault("text", text)
            chunk_kwargs.setdefault("ordinal", 0)
            chunk = Chunk(**chunk_kwargs)
            scored.append(ScoredChunk(chunk_id=cid, score=score, rank=rank, chunk=chunk))

        return scored

    def get_by_page(self, page_num: int, doc_id: Optional[str] = None) -> list[Chunk]:
        """Fetch all chunks belonging to a specific page number."""
        sql = "SELECT chunk_id, text, metadata_json FROM chunks_fts WHERE page_num = ?"
        params: list[object] = [page_num]
        if doc_id is not None:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY rowid ASC"

        cursor = self.con.execute(sql, params)
        chunks = []
        chunk_fields = {f for f in Chunk.__dataclass_fields__}
        for cid, text, meta_json in cursor.fetchall():
            payload = json.loads(meta_json) if meta_json else {}
            chunk_kwargs = {k: v for k, v in payload.items() if k in chunk_fields}
            chunk_kwargs.setdefault("chunk_id", cid)
            chunk_kwargs.setdefault("doc_id", cid.split("::")[0] if "::" in cid else cid)
            chunk_kwargs.setdefault("text", text)
            chunk_kwargs.setdefault("ordinal", 0)
            chunks.append(Chunk(**chunk_kwargs))
        return chunks

    def fuzzy_search_epic(
        self, target_epic: str, max_distance: int = 2, limit: int = 5
    ) -> list[ScoredChunk]:
        """Fuzzy search across voter chunks for EPIC IDs with up to max_distance edit distance."""
        clean_target = re.sub(r"[^A-Za-z0-9]", "", target_epic).upper()
        if len(clean_target) < 6:
            return []

        cursor = self.con.execute(
            "SELECT chunk_id, text, metadata_json FROM chunks_fts WHERE text LIKE '%EPIC:%'"
        )
        rows = cursor.fetchall()

        matches = []
        for cid, text, meta_json in rows:
            epics = re.findall(r"EPIC:\s*([A-Za-z0-9/]+)", text)
            best_dist = 999
            for ep in epics:
                clean_ep = re.sub(r"[^A-Za-z0-9]", "", ep).upper()
                if abs(len(clean_ep) - len(clean_target)) > max_distance:
                    continue
                dist = _levenshtein(clean_target, clean_ep)
                if dist < best_dist:
                    best_dist = dist

            if best_dist <= max_distance:
                payload = json.loads(meta_json) if meta_json else {}
                chunk_fields = {f for f in Chunk.__dataclass_fields__}
                chunk_kwargs = {k: v for k, v in payload.items() if k in chunk_fields}
                chunk_kwargs.setdefault("chunk_id", cid)
                chunk_kwargs.setdefault("doc_id", cid.split("::")[0] if "::" in cid else cid)
                chunk_kwargs.setdefault("text", text)
                chunk_kwargs.setdefault("ordinal", 0)
                chunk = Chunk(**chunk_kwargs)
                score = 1.0 - (best_dist * 0.05)
                matches.append((best_dist, ScoredChunk(chunk_id=cid, score=score, rank=1, chunk=chunk)))

        matches.sort(key=lambda x: x[0])
        return [m[1] for m in matches[:limit]]

    def close(self) -> None:
        """Close SQLite connection."""
        self.con.close()


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
