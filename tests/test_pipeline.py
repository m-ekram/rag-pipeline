"""Fast unit tests for the parts of ingestion that fail silently when wrong.

    pip install -r requirements-dev.txt
    python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.documents import Document

import config
from app.chunking import _heading_index, structured_split
from app.loaders import _strip_running_lines
from app.store import EmbedCache


def _keys(n: int, offset: int = 0) -> list[str]:
    return [f"{i + offset:040x}" for i in range(n)]


def test_embed_cache_roundtrip(tmp_path):
    cache = EmbedCache("m", root=tmp_path)
    vectors = np.arange(12, dtype=np.float32).reshape(4, 3)
    cache.add(_keys(4), vectors)

    reopened = EmbedCache("m", root=tmp_path)
    assert reopened.rows == 4
    np.testing.assert_array_equal(reopened.get(_keys(4)[::-1]), vectors[::-1])


def test_embed_cache_trims_torn_tail(tmp_path):
    cache = EmbedCache("m", root=tmp_path)
    cache.add(_keys(3), np.ones((3, 2), dtype=np.float32))
    # Simulate a kill after the vector write but before the key write.
    with (cache.dir / "vectors.f32").open("ab") as fh:
        fh.write(np.full((1, 2), 9, dtype=np.float32).tobytes())

    reopened = EmbedCache("m", root=tmp_path)
    assert reopened.rows == 3
    reopened.add(_keys(1, offset=100), np.full((1, 2), 5, dtype=np.float32))
    assert reopened.get(_keys(1, offset=100)).tolist() == [[5.0, 5.0]]
    assert EmbedCache("m", root=tmp_path).rows == 4


def test_embed_cache_rejects_dimension_change(tmp_path):
    cache = EmbedCache("m", root=tmp_path)
    cache.add(_keys(1), np.ones((1, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        cache.add(_keys(1, offset=5), np.ones((1, 4), dtype=np.float32))


def test_heading_index_skips_code_fences():
    text = "# Title\nintro\n```python\n# not a heading\n```\n## Section\nbody\n"
    offsets, paths = _heading_index(text)
    assert paths == ["Title", "Title > Section"]
    assert text[offsets[1]:].startswith("## Section")


def test_pdf_pages_merge_and_map_back(monkeypatch):
    monkeypatch.setattr(config, "MIN_CHUNK_CHARS", 1)
    line = "word " * 15
    pages = [
        Document(page_content="\n".join([f"p{n} {line}"] * 12), metadata={"source": "m.pdf", "page": n, "title": "M", "section": f"Ch {n}"})
        for n in (1, 2, 3)
    ]
    chunks = structured_split(pages, chunk_size=400, chunk_overlap=0, header_mode="path")

    assert any("page_end" in c.metadata for c in chunks), "chunks never cross a page break"
    for c in chunks:
        body = c.page_content.split("\n", 1)[1]
        assert body.startswith(f"p{c.metadata['page']} "), (c.metadata, body[:20])
        assert c.page_content.startswith(f"[M > Ch {c.metadata['page']} - p.{c.metadata['page']}]")


def test_path_clean_drops_headings_shared_across_documents(monkeypatch):
    monkeypatch.setattr(config, "MIN_CHUNK_CHARS", 1)
    body = "text " * 30
    # An intro under the H1 keeps each "##" section in its own chunk; otherwise
    # the splitter merges the H1 line into the first section's chunk.
    docs = [
        Document(
            page_content=f"# Page {i}\n{body}\n\n## Unique {i}\n{body}\n\n## Recap\n{body}",
            metadata={"source": f"p{i}.md", "title": f"file {i}"},
        )
        for i in range(3)
    ]
    chunks = structured_split(docs, chunk_size=200, chunk_overlap=0, header_mode="path-clean")
    headers = {c.page_content.split("\n", 1)[0] for c in chunks if c.metadata["source"] == "p0.md"}

    assert "[Page 0 > Unique 0]" in headers      # distinctive heading kept
    assert "[Page 0]" in headers                 # "Recap" (in 3 docs) dropped
    assert not any("Recap" in h for h in headers)
    assert any(c.metadata.get("section") == "Page 0 > Recap" for c in chunks)  # citation keeps it


class _ScoreByText:
    """Stand-in cross-encoder: fixed score per passage text."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score(self, pairs):
        return [self.scores[passage] for _, passage in pairs]


def test_rank_fusion_reranker_orders_by_fused_rank():
    from app.retriever import RankFusionReranker

    # First stage: a, b, c, d. Re-ranker strongly prefers d, then c.
    docs = [Document(page_content=t) for t in "abcd"]
    model = _ScoreByText({"a": 0.1, "b": 0.2, "c": 0.9, "d": 1.0})

    plain = RankFusionReranker(model=model, top_n=4, fuse=False).compress_documents(docs, "q")
    assert [d.page_content for d in plain] == ["d", "c", "b", "a"]

    fused = RankFusionReranker(model=model, top_n=4, fuse=True).compress_documents(docs, "q")
    # a = d = 1/61 + 1/64 (0.03202) > b = c = 1/62 + 1/63 (0.03200); ties keep
    # first-stage order.
    assert [d.page_content for d in fused] == ["a", "d", "b", "c"]

    # A candidate the re-ranker loves but the first stage ranked last does not
    # displace ones both rankings put near the top:
    # a = 1/61 + 1/63, b = 1/62 + 1/62, e = 1/65 + 1/61.
    docs5 = [Document(page_content=t) for t in "abcde"]
    model5 = _ScoreByText({"a": 0.5, "b": 0.8, "c": 0.1, "d": 0.2, "e": 1.0})
    top = RankFusionReranker(model=model5, top_n=2, fuse=True).compress_documents(docs5, "q")
    assert [d.page_content for d in top] == ["a", "b"]


def test_maxsim_matches_brute_force():
    from app.providers import maxsim

    rng = np.random.default_rng(0)
    q, d = rng.normal(size=(7, 16)), rng.normal(size=(23, 16))
    expected = sum(max(float(qi @ dj) for dj in d) for qi in q)
    assert maxsim(q, d) == pytest.approx(expected)


def test_fast_bm25_matches_rank_bm25():
    import random

    from rank_bm25 import BM25Okapi

    from app.bm25 import FastBM25Retriever

    rng = random.Random(3)
    vocab = [f"w{i}" for i in range(150)]
    weights = [1 / (i + 1) for i in range(150)]  # Zipf-like: some terms in most docs
    texts = [" ".join(rng.choices(vocab, weights, k=rng.randint(5, 60))) for _ in range(400)]
    docs = [Document(page_content=t, metadata={"i": i}) for i, t in enumerate(texts)]

    fast = FastBM25Retriever.from_documents(docs, k=10)
    reference = BM25Okapi([t.split() for t in texts])
    for _ in range(60):
        query = " ".join(rng.choices(vocab + ["unseen"], k=rng.randint(1, 8)))
        np.testing.assert_array_equal(fast.scores(query), reference.get_scores(query.split()))
        expected = [d.metadata["i"] for d in reference.get_top_n(query.split(), docs, n=10)]
        assert [d.metadata["i"] for d in fast.invoke(query)] == expected


def test_strip_running_lines_keeps_content():
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
    pages = [f"Manual - Chapter {n // 5}\nabout {words[n - 1]}\nthen {words[-n]} follows\n{n}" for n in range(1, 11)]
    cleaned = _strip_running_lines(pages)
    assert all("Manual" not in p for p in cleaned)
    assert all(p.splitlines()[-1].endswith("follows") for p in cleaned)
    assert cleaned[0] == "about alpha\nthen juliet follows"
