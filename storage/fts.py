"""SQLite FTS5 wrapper for lexical retrieval with slash-safe tokenization and parent context access."""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[\w/.-]+")


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    parent_context: str
    metadata: dict[str, Any]
    score: float = 0.0


class SQLiteFTS:
    """Persistent or in-memory SQLite FTS5 index for parent-child chunks."""

    def __init__(self, db_path: str = "data/db/fts.db"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create FTS5 virtual table with unicode61 tokenizer preserving slashes."""
        cur = self.con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
        if cur.fetchone():
            return
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
        self.con.commit()

    def count(self) -> int:
        cur = self.con.execute("SELECT COUNT(*) FROM chunks_fts")
        row = cur.fetchone()
        return row[0] if row else 0

    def index_chunks(self, chunks: Iterable[Any]) -> "SQLiteFTS":
        """Index chunks into the FTS5 table."""
        rows = []
        for c in chunks:
            meta = getattr(c, "metadata", {}) or {}
            page_num = meta.get("page", meta.get("page_num", 0))
            payload = c.to_payload() if hasattr(c, "to_payload") else {"text": c.text, "metadata": meta}
            rows.append(
                (
                    c.chunk_id,
                    getattr(c, "doc_id", "default"),
                    int(page_num) if str(page_num).isdigit() else 0,
                    c.text,
                    json.dumps(payload),
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
        return self

    def _format_query(self, query: str) -> str:
        tokens = _TOKEN_PATTERN.findall(query)
        if not tokens:
            return ""
        formatted = []
        for t in tokens:
            cleaned = t.strip("./-_")
            if not cleaned:
                continue
            if any(ch in t for ch in "/-_."):
                formatted.append(f'"{t}"')
            else:
                formatted.append(t)
        return " OR ".join(formatted) if formatted else ""

    def search(self, query: str, limit: int = 15) -> List[SearchResult]:
        """Search chunks by query and return SearchResult objects with parent_context."""
        fts_query = self._format_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT chunk_id, doc_id, page_num, text, metadata_json, bm25(chunks_fts) as rank_score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank_score ASC
            LIMIT ?
        """
        cur = self.con.execute(sql, (fts_query, limit))
        results = []
        for chunk_id, doc_id, page_num, text, metadata_json, score in cur.fetchall():
            meta = {}
            payload = {}
            if metadata_json:
                try:
                    payload = json.loads(metadata_json)
                    meta = payload.get("metadata", {})
                except Exception:
                    pass
            parent_ctx = (
                meta.get("parent_text")
                or meta.get("parent_context")
                or payload.get("parent_text")
                or payload.get("parent_context")
                or text
            )
            pg = payload.get("page") or payload.get("page_num") or meta.get("page") or meta.get("page_num") or page_num
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=int(pg) if str(pg).isdigit() else 0,
                    text=text,
                    parent_context=parent_ctx,
                    metadata=meta,
                    score=score,
                )
            )
        return results
