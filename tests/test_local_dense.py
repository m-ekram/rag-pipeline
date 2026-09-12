"""The Qdrant-free vector index: ranking, persistence and page lookups.

The embedder is a deterministic keyword counter, so no model is loaded.
"""

import numpy as np

from ingestion.documents import Chunk
from retrieval import dense
from retrieval.local_dense import LocalDenseIndex


class _KeywordEmbedder:
    """Three-dimensional "embeddings": one axis per keyword."""

    AXES = ("tax", "roll", "plan")

    def _vector(self, text):
        return np.array([text.lower().count(k) for k in self.AXES], dtype="float32") + 1e-3

    def embed_passages(self, texts, show_progress=False):
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


def _chunks():
    return [
        Chunk(chunk_id="a::0", doc_id="pmp#p66", text="land use plan plan", ordinal=0, page=66, title="pmp"),
        Chunk(chunk_id="b::0", doc_id="roll#p66", text="voter roll roll", ordinal=0, page=66, title="roll"),
        Chunk(chunk_id="c::0", doc_id="tax#p2", text="tax tax income", ordinal=0, page=2),
    ]


def _index(tmp_path):
    index = LocalDenseIndex("c", _KeywordEmbedder(), root=tmp_path)
    index.index(_chunks())
    return index


def test_search_ranks_by_cosine_similarity(tmp_path):
    hits = _index(tmp_path).search("roll", limit=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "b::0"
    assert hits[0].chunk.text == "voter roll roll"
    assert hits[0].score >= hits[1].score
    assert [h.rank for h in hits] == [1, 2]


def test_index_persists_across_processes(tmp_path):
    _index(tmp_path)
    reopened = LocalDenseIndex("c", _KeywordEmbedder(), root=tmp_path)
    assert reopened.exists() and reopened.count() == 3
    assert reopened.search("tax", limit=1)[0].chunk_id == "c::0"


def test_get_by_page_can_filter_to_one_document(tmp_path):
    index = _index(tmp_path)
    assert {c.chunk_id for c in index.get_by_page(66)} == {"a::0", "b::0"}
    assert [c.chunk_id for c in index.get_by_page(66, doc_id="pmp#p66")] == ["a::0"]


def test_read_only_views_follow_index_order(tmp_path):
    index = _index(tmp_path)
    assert index.chunk_ids == ["a::0", "b::0", "c::0"]
    assert index.matrix.shape == (3, 3)
    assert np.allclose(np.linalg.norm(index.matrix, axis=1), 1.0, atol=1e-3)


def test_recreate_empties_the_index(tmp_path):
    index = _index(tmp_path)
    index.recreate()
    assert not index.exists()
    assert index.search("tax") == []
    assert not (tmp_path / "c").exists()


def test_inconsistent_files_read_as_no_index(tmp_path):
    """A crash between the two file replaces must not load a mismatched index."""
    _index(tmp_path)
    (tmp_path / "c" / "payloads.jsonl").write_text(
        '{"chunk_id": "a::0", "doc_id": "d", "text": "t", "ordinal": 0}\n', encoding="utf-8"
    )
    assert not LocalDenseIndex("c", _KeywordEmbedder(), root=tmp_path).exists()


def test_unusable_qdrant_falls_back_to_the_local_index(monkeypatch):
    monkeypatch.delenv("RAG_VECTOR_STORE", raising=False)
    monkeypatch.setitem(dense._QDRANT_CLIENTS, "http://qdrant.invalid:1", None)
    index = dense.make_dense_index("probe_only", _KeywordEmbedder(), url="http://qdrant.invalid:1")
    assert isinstance(index, LocalDenseIndex)
