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


def test_prompts_for_follows_model_cards():
    from app.providers import prompts_for

    assert prompts_for("BAAI/bge-small-en-v1.5") == (config.BGE_QUERY_PROMPT, "")
    assert prompts_for("nomic-ai/nomic-embed-text-v1.5") == ("search_query: ", "search_document: ")
    assert prompts_for("snowflake/snowflake-arctic-embed-m")[0].startswith("Represent this sentence")
    assert prompts_for("sentence-transformers/all-MiniLM-L6-v2") == ("", "")


def test_splade_scores_equal_brute_force_dot_product():
    from app.splade import SpladeRetriever

    rng = np.random.default_rng(1)
    vocab = 300

    def sparse(nnz):
        idx = rng.choice(vocab, size=nnz, replace=False)
        return idx.astype(np.int64), rng.random(nnz).astype(np.float32)

    vectors = [sparse(rng.integers(5, 40)) for _ in range(50)]
    docs = [Document(page_content=str(i), metadata={"i": i}) for i in range(50)]
    query = sparse(12)

    class FakeEncoder:
        def encode_query(self, text):
            return query

    retriever = SpladeRetriever.from_vectors(docs, vectors, FakeEncoder(), k=5)

    dense_q = np.zeros(vocab, dtype=np.float32)
    dense_q[query[0]] = query[1]
    expected = []
    for idx, val in vectors:
        d = np.zeros(vocab, dtype=np.float32)
        d[idx] = val
        expected.append(float(dense_q @ d))
    np.testing.assert_allclose(retriever.scores("q"), expected, rtol=1e-5)
    assert [d.metadata["i"] for d in retriever.invoke("q")] == list(np.argsort(expected)[::-1][:5])


def test_maxsim_matches_brute_force():
    from app.providers import maxsim

    rng = np.random.default_rng(0)
    q, d = rng.normal(size=(7, 16)), rng.normal(size=(23, 16))
    expected = sum(max(float(qi @ dj) for dj in d) for qi in q)
    assert maxsim(q, d) == pytest.approx(expected)


def test_header_target_sparse_embeds_body_only(monkeypatch, tmp_path):
    from app.store import build_index, save_index, load_chunks

    monkeypatch.setattr(config, "MIN_CHUNK_CHARS", 1)
    monkeypatch.setattr(config, "EMBED_CACHE", False)
    body = "text " * 30
    docs = [Document(page_content=f"# Page\n{body}\n\n## Part\n{body}", metadata={"source": "p.md", "title": "p"})]
    chunks = structured_split(docs, chunk_size=200, chunk_overlap=0, header_mode="path-clean", header_target="sparse")

    embedded: list[str] = []

    class Recorder:
        def embed_documents(self, texts):
            embedded.extend(texts)
            return [[float(len(t)), 1.0] for t in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0]

    store = build_index(chunks, show_progress=False, embeddings=Recorder(), model_name="recorder")
    assert embedded and not any(t.startswith("[") for t in embedded)          # dense: body only
    assert all(c.page_content.startswith("[") for c in chunks)                # chunk keeps its header
    stored = list(store.docstore._dict.values())
    assert all(d.page_content.startswith("[") and "embed_text" not in d.metadata for d in stored)

    save_index(store, chunks, str(tmp_path))
    assert all("embed_text" not in c.metadata for c in load_chunks(str(tmp_path)))


class _FakeGenerator:
    """Deterministic stand-in for the local LLM."""

    def __init__(self):
        self.calls = 0

    def chat(self, prompt, max_new_tokens=128):
        self.calls += 1
        if "passage (2-3 sentences)" in prompt:
            return "Use app.mount with StaticFiles to serve a directory."
        if "Rewrite this question" in prompt:
            return "1. How to host stylesheets?\n2. Serving pictures from a folder\n- how to host stylesheets?"
        return "1. How do I host my stylesheets and pictures?\n- Where do browser assets get served from?\nWhat is a mount?"


def test_parse_questions_strips_numbering_and_duplicates():
    from app.doc2query import parse_questions

    assert parse_questions("1. How do I x?\n- how do I X?\n* Another question here\nshort", 5) == [
        "How do I x?",
        "Another question here",
    ]
    assert len(parse_questions("q one is long\nq two is long\nq three is long", 2)) == 2


def test_doc2query_expansion_is_indexed_but_never_scored(monkeypatch, tmp_path):
    from app.bm25 import FastBM25Retriever
    from app.doc2query import expand_chunks
    from app.store import build_index
    from eval.evaluate import is_relevant

    monkeypatch.setattr(config, "EMBED_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "EMBED_CACHE", False)
    chunks = [
        Document(page_content="[Static Files]\nMount a directory with StaticFiles.", metadata={"source": "s.md", "chunk_id": "a"}),
        Document(page_content="[Testing]\nUse TestClient to call endpoints.", metadata={"source": "t.md", "chunk_id": "b"}),
    ]
    generator = _FakeGenerator()
    expand_chunks(chunks, generator=generator, n=2, show_progress=False)

    assert chunks[0].metadata["expansion"].startswith("How do I host my stylesheets")
    assert chunks[0].page_content == "[Static Files]\nMount a directory with StaticFiles."  # returned text untouched

    # The expansion reaches the lexical index: "stylesheets" only occurs there.
    assert FastBM25Retriever.from_documents(chunks, k=1).invoke("stylesheets")[0].metadata["chunk_id"] == "a"

    # ...and the dense one.
    embedded: list[str] = []

    class Recorder:
        def embed_documents(self, texts):
            embedded.extend(texts)
            return [[1.0, float(i)] for i, _ in enumerate(texts)]

        def embed_query(self, text):
            return [1.0, 0.0]

    build_index(chunks, show_progress=False, embeddings=Recorder(), model_name="recorder")
    assert "stylesheets" in embedded[0]

    # Generated text can never make a chunk count as relevant.
    assert not is_relevant(chunks[1], {"relevant_sources": ["t.md"], "must_contain": ["stylesheets"]})

    # Cached: a second expansion of the same bodies generates nothing.
    calls = generator.calls
    expand_chunks([Document(page_content=c.page_content, metadata={}) for c in chunks], generator=generator, n=2, show_progress=False)
    assert generator.calls == calls


def test_rewrite_fusion_ranks_documents_found_by_several_variants(monkeypatch, tmp_path):
    from app.query_rewrite import RewriteFusionRetriever, query_variants

    monkeypatch.setattr(config, "EMBED_CACHE_DIR", str(tmp_path))
    generator = _FakeGenerator()
    variants = query_variants("How do I serve CSS?", generator)
    assert variants[0] == "How do I serve CSS?"
    assert "How to host stylesheets?" in variants and variants[-1].startswith("Use app.mount")
    assert len(variants) == 1 + 2 + 1  # original + 2 unique paraphrases + passage

    docs = {name: Document(page_content=name, metadata={"chunk_id": name}) for name in "abcd"}

    class Base:
        def invoke(self, query):
            # Only the original query ranks "a" first; every rewrite finds "c".
            return [docs["a"], docs["b"]] if query == "How do I serve CSS?" else [docs["c"], docs["d"]]

    fused = RewriteFusionRetriever(base=Base(), k=2, generator=generator).invoke("How do I serve CSS?")
    assert [d.metadata["chunk_id"] for d in fused] == ["c", "d"]
    calls = generator.calls
    query_variants("How do I serve CSS?", generator)  # cached
    assert generator.calls == calls


def test_crossval_selects_on_one_set_and_scores_on_the_other():
    from eval.evaluate import crossval

    def result(name, dev_hits, v1_hits):
        detail = [{"set": "dev", "hit": i < dev_hits, "first_relevant_rank": 1 if i < dev_hits else 0} for i in range(4)]
        detail += [{"set": "heldout", "hit": i < v1_hits, "first_relevant_rank": 1 if i < v1_hits else 0} for i in range(4)]
        return {"config": name, "detail": detail}

    # "fit-dev" is best on dev but poor on held-out; "general" is best on held-out.
    results = [result("tuned", 2, 2), result("fit-dev", 4, 1), result("general", 3, 3)]
    cv = crossval(results, ["dev", "heldout"])

    assert cv["folds"][0]["chosen"] == "fit-dev" and cv["folds"][0]["hits"] == 1  # chosen on dev, scored on v1
    assert cv["folds"][1]["chosen"] == "general" and cv["folds"][1]["hits"] == 3  # chosen on v1, scored on dev
    assert cv["hits"] == 4 and cv["n"] == 8 and cv["honest_estimate"] == 0.5


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
