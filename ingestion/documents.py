"""Core data types shared by ingestion, retrieval and generation.

`Chunk` carries the metadata needed for citations (`[Document X, Section Y,
Page Z]`) all the way through the pipeline. Phase 1 of the plan calls this out
as don't-skip: metadata dropped at chunk time cannot be recovered downstream.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass(frozen=True)
class Document:
    """A source document, before chunking."""

    doc_id: str
    text: str
    title: str = ""
    source: str = ""          # provenance: file path, URL, or dataset name
    page: Optional[int] = None    # 1-indexed, for page-oriented sources (PDF)
    section: Optional[str] = None  # nearest enclosing heading, for structured sources
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text, with a pointer back to where it came from."""

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int              # position of this chunk within its document, 0-indexed
    title: str = ""
    source: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    start_word: int = 0       # word offset into the parent document
    end_word: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Flat dict suitable for a Qdrant point payload."""
        payload = asdict(self)
        if self.page is not None:
            payload["page_num"] = self.page
        elif "page" in self.metadata:
            try:
                payload["page_num"] = int(self.metadata["page"])
            except (ValueError, TypeError):
                pass
        if "block_type" in self.metadata:
            payload["block_type"] = self.metadata["block_type"]
        if "parent_id" in self.metadata:
            payload["parent_id"] = self.metadata["parent_id"]
        # Qdrant payloads are flat key/value; keep nested metadata but drop it
        # when empty so payloads stay readable in the dashboard.
        if not payload["metadata"]:
            payload.pop("metadata")
        return payload

    def citation(self) -> str:
        """Human-readable provenance string for grounded answers."""
        parts = [f"Document {self.title or self.doc_id}"]
        if self.section:
            parts.append(f"Section {self.section}")
        if self.page is not None:
            parts.append(f"Page {self.page}")
        return "[" + ", ".join(parts) + "]"
