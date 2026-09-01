"""The eval scorer, and the integrity of the checked-in golden set.

If `is_relevant` is wrong, every published number is wrong in a way no amount of
re-running detects. The `must_contain` condition is the part that matters most:
without it a 40-page document scores a hit merely for being the right file.
"""

import pytest
from langchain_core.documents import Document

from eval.evaluate import ABLATIONS, CONFIGS, is_relevant, load_questions, validate


def chunk(text, source):
    return Document(page_content=text, metadata={"source": source})


def question(sources, must_contain=None):
    q = {"question": "does not matter here", "relevant_sources": sources}
    if must_contain is not None:
        q["must_contain"] = must_contain
    return q


# -- source matching -----------------------------------------------------------


def test_the_right_source_with_no_must_contain_is_a_hit():
    assert is_relevant(chunk("anything", "fastapi/tutorial__testing.md"),
                       question(["tutorial__testing.md"]))


def test_the_wrong_source_is_never_a_hit():
    assert not is_relevant(chunk("anything", "fastapi/tutorial__cors.md"),
                           question(["tutorial__testing.md"]))


def test_sources_match_on_a_path_suffix():
    """Golden entries name a filename, not the full corpus-relative path."""
    assert is_relevant(chunk("x", "deep/nested/tutorial__testing.md"),
                       question(["tutorial__testing.md"]))


def test_any_of_several_listed_sources_counts():
    q = question(["tutorial__security__first-steps.md", "tutorial__security__index.md"])
    assert is_relevant(chunk("x", "fastapi/tutorial__security__index.md"), q)


def test_a_chunk_with_no_source_metadata_is_not_a_hit():
    assert not is_relevant(Document(page_content="x", metadata={}),
                           question(["tutorial__testing.md"]))


# -- the must_contain condition ------------------------------------------------


def test_must_contain_present_is_a_hit():
    assert is_relevant(chunk("use TestClient to call it", "a/tutorial__testing.md"),
                       question(["tutorial__testing.md"], ["TestClient"]))


def test_right_file_but_missing_the_phrase_is_a_miss():
    """This is the whole point of must_contain."""
    assert not is_relevant(chunk("unrelated prose about deployment", "a/tutorial__testing.md"),
                           question(["tutorial__testing.md"], ["TestClient"]))


def test_must_contain_is_case_insensitive():
    assert is_relevant(chunk("use testclient here", "a/tutorial__testing.md"),
                       question(["tutorial__testing.md"], ["TestClient"]))


def test_any_one_of_several_phrases_is_enough():
    q = question(["tutorial__request-files.md"], ["UploadFile", "File("])
    assert is_relevant(chunk("declare it with File(...)", "a/tutorial__request-files.md"), q)


def test_an_empty_must_contain_list_does_not_gate():
    assert is_relevant(chunk("anything", "a/tutorial__testing.md"),
                       question(["tutorial__testing.md"], []))


def test_the_phrase_alone_is_not_enough_without_the_right_file():
    """Both conditions must hold, not either."""
    assert not is_relevant(chunk("use TestClient to call it", "a/tutorial__cors.md"),
                           question(["tutorial__testing.md"], ["TestClient"]))


# -- validate ------------------------------------------------------------------


DOCS = [
    Document(
        page_content="Use TestClient to exercise the endpoints before shipping them. " * 4,
        metadata={"source": "tutorial__testing.md", "title": "Testing"},
    )
]


def test_validate_passes_a_reachable_question(capsys):
    assert validate(DOCS, [question(["tutorial__testing.md"], ["TestClient"])]) == 0


def test_validate_flags_a_source_that_does_not_exist(capsys):
    assert validate(DOCS, [question(["no-such-file.md"])]) == 1
    assert "UNREACHABLE" in capsys.readouterr().out


def test_validate_flags_a_phrase_that_appears_nowhere(capsys):
    """A question that can never be scored a hit would silently drag the number down."""
    assert validate(DOCS, [question(["tutorial__testing.md"], ["NoSuchSymbol"])]) == 1
    assert "UNMATCHED" in capsys.readouterr().out


def test_validate_counts_every_problem(capsys):
    problems = validate(DOCS, [question(["nope.md"]),
                               question(["tutorial__testing.md"], ["NoSuchSymbol"])])
    assert problems == 2


# -- the checked-in golden set -------------------------------------------------


def test_the_golden_set_loads():
    assert len(load_questions()) >= 30


def test_every_golden_question_has_the_required_fields():
    for q in load_questions():
        assert q.get("question"), q
        assert q.get("relevant_sources"), q


def test_golden_questions_are_unique():
    texts = [q["question"] for q in load_questions()]
    assert len(texts) == len(set(texts))


def test_every_golden_question_is_tagged_with_a_known_failure_class():
    """The three classes break for different reasons; an untagged question hides that."""
    for q in load_questions():
        assert set(q.get("tags", [])) <= {"exact", "paraphrase", "multihop"}, q
        assert q.get("tags"), q


def test_all_three_failure_classes_are_represented():
    tags = {t for q in load_questions() for t in q["tags"]}
    assert tags == {"exact", "paraphrase", "multihop"}


# -- harness configuration -----------------------------------------------------


def test_the_baseline_is_genuinely_unoptimised():
    """If the baseline drifts toward the tuned config, the comparison means nothing."""
    baseline = CONFIGS["baseline"]
    assert baseline["splitter"] == "naive"
    assert baseline["chunk_overlap"] == 0
    assert baseline["headers"] is False
    assert baseline["hybrid"] is False


def test_the_tuned_config_is_what_ships():
    tuned = CONFIGS["tuned"]
    assert tuned["splitter"] == "structured"
    assert tuned["headers"] is True
    assert tuned["hybrid"] is True


def test_the_ablation_grid_covers_both_retrieval_legs():
    assert {"dense-only", "hybrid"} <= set(ABLATIONS)
