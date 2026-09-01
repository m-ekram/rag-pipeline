"""Computed citation attribution.

This is the feature that carries citations when the chat model will not emit
them (verified: local Qwen2.5-3B produces zero markers unaided). Because a
computed citation is a weaker claim than a model-asserted one, the conservative
behaviours are the ones worth pinning: no confident match means no marker, and
code is never annotated.

A stub embedder keeps these tests offline and instant.
"""

import pytest
from langchain_core.documents import Document

import config
from app.attribution import _cosine, _split_outside_code, attribute

ALPHA, BETA, OTHER = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


class StubEmbeddings:
    """Maps a keyword to a basis vector, so similarity is exactly controllable."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("embedding backend is down")
        self.calls.append(list(texts))
        out = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                out.append(ALPHA)
            elif "beta" in lowered:
                out.append(BETA)
            else:
                out.append(OTHER)
        return out


DOCS = [Document(page_content="alpha passage"), Document(page_content="beta passage")]

# Comfortably past CITE_MIN_CHARS (40) so the length guard is not what is tested.
ALPHA_SENTENCE = "The alpha subsystem retries three times before failing."
BETA_SENTENCE = "The beta scheduler drains its queue on shutdown."
UNRELATED = "Nothing in this sentence resembles the indexed passages at all."


def attributed(answer, docs=DOCS, embeddings=None):
    return attribute(answer, docs, embeddings or StubEmbeddings())


# -- cosine --------------------------------------------------------------------


def test_cosine_of_identical_vectors_is_one():
    assert _cosine(ALPHA, ALPHA) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert _cosine(ALPHA, BETA) == pytest.approx(0.0)


def test_cosine_of_a_zero_vector_is_zero_not_a_crash():
    assert _cosine([0.0, 0.0, 0.0], ALPHA) == 0.0


# -- when attribution should do nothing at all ---------------------------------


def test_an_answer_that_already_cites_is_left_alone():
    """The model's own judgement outranks ours."""
    answer = f"{ALPHA_SENTENCE} [2]"
    assert attributed(answer) == answer


def test_no_retrieved_documents_means_no_markers():
    assert attributed(ALPHA_SENTENCE, docs=[]) == ALPHA_SENTENCE


def test_an_empty_answer_is_returned_unchanged():
    assert attributed("   ") == "   "


def test_an_embedding_failure_degrades_to_the_plain_answer():
    """A broken embedder must cost citations, not the answer."""
    assert attributed(ALPHA_SENTENCE, embeddings=StubEmbeddings(fail=True)) == ALPHA_SENTENCE


# -- matching ------------------------------------------------------------------


def test_a_sentence_matching_passage_one_is_marked_one():
    assert attributed(ALPHA_SENTENCE) == "The alpha subsystem retries three times before failing [1]."


def test_a_sentence_matching_passage_two_is_marked_two():
    assert attributed(BETA_SENTENCE).endswith("[2].")


def test_a_sentence_below_the_similarity_floor_gets_no_marker():
    assert attributed(UNRELATED) == UNRELATED


def test_the_marker_goes_before_the_trailing_punctuation():
    assert " [1]." in attributed(ALPHA_SENTENCE)
    assert "[1] ." not in attributed(ALPHA_SENTENCE)


def test_each_sentence_on_a_line_is_matched_independently():
    out = attributed(f"{ALPHA_SENTENCE} {BETA_SENTENCE}")
    assert "[1]" in out and "[2]" in out


def test_a_short_sentence_is_never_marked():
    """Below CITE_MIN_CHARS there is not enough signal to attribute honestly."""
    short = "Alpha is fast."
    assert len(short) < config.CITE_MIN_CHARS
    assert attributed(short) == short


# -- what must never be annotated ----------------------------------------------


def test_split_outside_code_flags_fenced_blocks():
    parts = _split_outside_code("prose\n```\ncode\n```\nmore")
    assert [is_code for _, is_code in parts] == [False, True, False]


def test_fenced_code_is_never_annotated():
    answer = f"{ALPHA_SENTENCE}\n```python\nalpha_client.retry(times=3, backoff=True)\n```"
    out = attributed(answer)
    code = out.split("```")[1]
    assert "[1]" not in code
    assert "[1]" in out.split("```")[0]


def test_the_code_fence_itself_survives_round_tripping():
    answer = f"{ALPHA_SENTENCE}\n```python\nalpha_client.retry()\n```\n{BETA_SENTENCE}"
    out = attributed(answer)
    assert out.count("```") == 2
    assert "alpha_client.retry()" in out


def test_headings_are_not_annotated():
    answer = f"## The alpha subsystem and its retry behaviour explained\n{ALPHA_SENTENCE}"
    heading = attributed(answer).split("\n")[0]
    assert "[1]" not in heading


def test_table_rows_are_not_annotated():
    answer = "| alpha subsystem | retries three times before it finally fails |"
    assert attributed(answer) == answer


def test_citation_mode_is_auto_for_the_local_provider():
    """Small local models do not cite, so auto must be the local default."""
    assert config.CITATION_MODE in {"auto", "model", "off"}
