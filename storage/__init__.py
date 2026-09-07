"""Storage modules for local persistent indexing."""

from .fts import SQLiteFTS, SearchResult

__all__ = ["SQLiteFTS", "SearchResult"]
