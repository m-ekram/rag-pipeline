"""Offline tests for cleaning, dedup, chunking and loaders."""

import json

import pytest

from ingestion.chunking import (
    FixedSizeChunker,
    SentenceAwareChunker,
    chunk_documents,
    split_sentences,
)
from ingestion.clean import clean_text
from ingestion.dedup import dedup_documents, fingerprint
from ingestion.documents import Chunk, Document
from ingestion.loaders import load_beir_corpus, load_html, load_qrels


# --- cleaning -----------------------------------------------------------


def test_clean_joins_pdf_hyphen_wrapping():
    assert clean_text("inter-\nnational trade") == "international trade"


def test_clean_joins_wrapped_lines_but_keeps_paragraphs():
    assert clean_text("a sentence that\nwraps mid-clause") == "a sentence that wraps mid-clause"
    assert clean_text("Para one.\n\nPara two.") == "Para one.\n\nPara two."


def test_clean_normalises_ligatures_and_control_chars():
    assert clean_text("ﬁnance\x00 report") == "finance report"


def test_clean_handles_empty():
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""


# --- dedup --------------------------------------------------------------


def test_fingerprint_ignores_case_punctuation_whitespace():
    assert fingerprint("Hello, World!") == fingerprint("hello   world")
    assert fingerprint("hello world") != fingerprint("goodbye world")


def test_dedup_keeps_first_occurrence_and_preserves_order():
    docs = [
        Document(doc_id="a", text="unique one"),
        Document(doc_id="b", text="shared text"),
        Document(doc_id="c", text="Shared  TEXT!"),
        Document(doc_id="d", text="unique two"),
    ]
    kept = list(dedup_documents(docs))
    assert [d.doc_id for d in kept] == ["a", "b", "d"]


# --- sentence splitting -------------------------------------------------


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_sentences_respects_abbreviations():
    # "Dr. Smith" must not become two sentences.
    assert split_sentences("Dr. Smith paid $5. Then he left.") == [
        "Dr. Smith paid $5.",
        "Then he left.",
    ]


def test_split_sentences_treats_paragraphs_as_boundaries():
    assert split_sentences("no punctuation\n\nsecond para") == [
        "no punctuation",
        "second para",
    ]


# --- chunking -----------------------------------------------------------


def _doc(text, **kw):
    return Document(doc_id="d1", text=text, **kw)


def test_fixed_chunker_splits_on_word_count():
    doc = _doc(" ".join(str(i) for i in range(10)))
    chunks = list(FixedSizeChunker(chunk_size=4).chunk(doc))
    assert [c.text.split() for c in chunks] == [
        ["0", "1", "2", "3"], ["4", "5", "6", "7"], ["8", "9"],
    ]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_fixed_chunker_loses_no_words():
    doc = _doc(" ".join(str(i) for i in range(97)))
    chunks = list(FixedSizeChunker(chunk_size=10).chunk(doc))
    recovered = " ".join(c.text for c in chunks).split()
    assert recovered == doc.text.split()


def test_fixed_chunker_overlap_repeats_words():
    doc = _doc(" ".join(str(i) for i in range(10)))
    chunks = list(FixedSizeChunker(chunk_size=5, overlap=2).chunk(doc))
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_fixed_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=5, overlap=5)
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0)


def test_chunker_propagates_citation_metadata():
    doc = _doc("some text here", title="Annual Report", source="/x.pdf",
               page=7, section="Risk Factors")
    chunk = next(iter(FixedSizeChunker(chunk_size=10).chunk(doc)))
    assert chunk.page == 7
    assert chunk.section == "Risk Factors"
    assert chunk.citation() == "[Document Annual Report, Section Risk Factors, Page 7]"


def test_citation_falls_back_to_doc_id_when_untitled():
    # FiQA has no titles at all, so this is the common path for that corpus.
    chunk = Chunk(chunk_id="3::0", doc_id="3", text="t", ordinal=0)
    assert chunk.citation() == "[Document 3]"


def test_sentence_chunker_never_splits_a_sentence():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    chunks = list(SentenceAwareChunker(max_words=6).chunk(_doc(text)))
    for chunk in chunks:
        assert chunk.text.endswith(".")
    assert " ".join(c.text for c in chunks) == text


def test_sentence_chunker_emits_oversized_sentence_alone_rather_than_truncating():
    long_sentence = " ".join(["word"] * 50) + "."
    chunks = list(SentenceAwareChunker(max_words=10, hard_max_words=100).chunk(
        _doc(long_sentence)))
    assert len(chunks) == 1
    assert len(chunks[0].text.split()) == 50  # nothing dropped


def test_sentence_chunker_force_splits_past_hard_cap_without_losing_words():
    """Real corpora contain unpunctuated walls of text; those must not exceed
    the embedding model's token limit or their tails vanish at encode time."""
    wall = " ".join(["word"] * 500) + "."
    chunks = list(SentenceAwareChunker(max_words=100, hard_max_words=150).chunk(
        _doc(wall)))
    assert len(chunks) == 4
    assert all(len(c.text.split()) <= 150 for c in chunks)
    assert len(" ".join(c.text for c in chunks).split()) == 500
    assert [c.ordinal for c in chunks] == [0, 1, 2, 3]


def test_sentence_chunker_flushes_buffer_before_force_splitting():
    # The wall must start with a capital, or the splitter correctly reads the
    # whole thing as one sentence (a lowercase continuation is not a boundary).
    text = "Short one. " + " ".join(["Word"] + ["word"] * 299) + "."
    chunks = list(SentenceAwareChunker(max_words=50, hard_max_words=100).chunk(
        _doc(text)))
    assert chunks[0].text == "Short one."
    assert all(len(c.text.split()) <= 100 for c in chunks)
    assert len(" ".join(c.text for c in chunks).split()) == 302


def test_sentence_chunker_rejects_hard_cap_below_max():
    with pytest.raises(ValueError):
        SentenceAwareChunker(max_words=200, hard_max_words=100)


def test_empty_document_yields_no_chunks():
    assert list(FixedSizeChunker().chunk(_doc("   "))) == []
    assert list(SentenceAwareChunker().chunk(_doc(""))) == []


def test_chunk_ids_are_unique_and_stable():
    docs = [_doc("a b c d e f"), Document(doc_id="d2", text="g h i")]
    chunks = list(chunk_documents(docs, FixedSizeChunker(chunk_size=2)))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "d1::0"


# --- loaders ------------------------------------------------------------


def test_load_beir_corpus_reads_id_text_title(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"_id": "3", "title": "", "text": "hello", "metadata": {}}) + "\n"
        + json.dumps({"_id": "4", "title": "T", "text": "world", "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    docs = list(load_beir_corpus(str(path)))
    assert [(d.doc_id, d.text, d.title) for d in docs] == [
        ("3", "hello", ""), ("4", "world", "T"),
    ]
    assert all(d.source == "fiqa" for d in docs)


def test_load_beir_corpus_skips_malformed_lines(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"_id": "1", "text": "ok"}) + "\n{ not json\n"
        + json.dumps({"_id": "2", "text": "also ok"}) + "\n",
        encoding="utf-8",
    )
    assert [d.doc_id for d in load_beir_corpus(str(path))] == ["1", "2"]


def test_load_qrels_skips_header_and_parses_scores(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text("query-id\tcorpus-id\tscore\n8\t566392\t1\n8\t65404\t1\n9\t111\t1\n",
                    encoding="utf-8")
    qrels = load_qrels(str(path))
    assert qrels == {"8": {"566392": 1, "65404": 1}, "9": {"111": 1}}


def test_load_qrels_tolerates_missing_header(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text("8\t566392\t1\n", encoding="utf-8")
    assert load_qrels(str(path)) == {"8": {"566392": 1}}


def test_load_html_sections_populate_citation_metadata():
    html = """
    <html><head><title>Doc</title></head><body>
    <h2>Risk Factors</h2><p>Markets may fall.</p>
    <h2>Outlook</h2><p>Things may improve.</p>
    <script>ignored()</script>
    </body></html>
    """
    docs = list(load_html(html, is_markup=True))
    assert [d.section for d in docs] == ["Risk Factors", "Outlook"]
    assert "Markets may fall." in docs[0].text
    assert "ignored" not in " ".join(d.text for d in docs)


# --- corpus assembly / contamination ------------------------------------


@pytest.fixture
def tiny_corpus(tmp_path):
    """3 judged docs + 20 unjudged, so contamination levels are checkable."""
    corpus = tmp_path / "corpus.jsonl"
    lines = [json.dumps({"_id": f"j{i}", "title": "", "text": f"judged doc {i} content"})
             for i in range(3)]
    lines += [json.dumps({"_id": f"u{i}", "title": "", "text": f"unjudged doc {i} content"})
              for i in range(20)]
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")

    qrels = tmp_path / "test.tsv"
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(f"q{i}\tj{i}\t1\n" for i in range(3)),
        encoding="utf-8",
    )
    return str(corpus), str(qrels)


def test_build_corpus_zero_distractors_keeps_only_judged(tiny_corpus):
    from ingestion.pipeline import build_corpus

    corpus, qrels = tiny_corpus
    docs = build_corpus(corpus_path=corpus, qrels_path=qrels, in_domain_distractors=0)
    assert sorted(d.doc_id for d in docs) == ["j0", "j1", "j2"]


def test_build_corpus_always_keeps_every_judged_doc(tiny_corpus):
    """A judged doc sampled away would make its query silently unanswerable."""
    from ingestion.pipeline import build_corpus

    corpus, qrels = tiny_corpus
    for n in (0, 3, 9, None):
        docs = build_corpus(corpus_path=corpus, qrels_path=qrels,
                            in_domain_distractors=n)
        assert {"j0", "j1", "j2"} <= {d.doc_id for d in docs}


def test_build_corpus_hits_target_contamination(tiny_corpus):
    from ingestion.pipeline import build_corpus, contamination_ratio

    corpus, qrels = tiny_corpus
    judged = {"j0", "j1", "j2"}
    docs = build_corpus(corpus_path=corpus, qrels_path=qrels, in_domain_distractors=3)
    assert len(docs) == 6
    assert contamination_ratio(docs, judged) == pytest.approx(0.5)


def test_build_corpus_is_reproducible_under_seed(tiny_corpus):
    from ingestion.pipeline import build_corpus

    corpus, qrels = tiny_corpus
    kw = dict(corpus_path=corpus, qrels_path=qrels, in_domain_distractors=5)
    first = [d.doc_id for d in build_corpus(seed=7, **kw)]
    assert first == [d.doc_id for d in build_corpus(seed=7, **kw)]
    assert first != [d.doc_id for d in build_corpus(seed=8, **kw)]


def test_build_corpus_caps_at_available_distractors(tiny_corpus):
    from ingestion.pipeline import build_corpus

    corpus, qrels = tiny_corpus
    docs = build_corpus(corpus_path=corpus, qrels_path=qrels,
                        in_domain_distractors=10_000)
    assert len(docs) == 23  # 3 judged + all 20 unjudged, not an error
