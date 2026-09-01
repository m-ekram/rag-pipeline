"""The eval harness: metric maths, variant isolation, and the baseline gate.

The regression tests here exist for a specific bug. Ablation variants used to
take their MMR lambda from ambient config when their spec did not name one:

    config.MMR_LAMBDA = spec.get("lambda", 1.0 if not spec["mmr"] else original[1])

Once MMR_LAMBDA defaulted to 1.0, every "+mmr" variant silently ran with MMR
disabled and scored identically to its non-MMR twin, erasing the finding the
ablation table exists to document. Nothing failed; the table just quietly became
a lie. These tests make that failure mode loud.
"""

import json

import pytest
from langchain_core.documents import Document

import config
from app.retriever import build_retriever, dense_only_retriever
from eval.evaluate import (
    ABLATIONS,
    CONFIGS,
    METRICS,
    by_tag,
    check_against_baseline,
    retriever_for,
    score,
    summarise,
)


class StubStore:
    """Records how the dense retriever was configured."""

    def __init__(self):
        self.search_type = None
        self.search_kwargs = None

    def as_retriever(self, search_type=None, search_kwargs=None):
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        return f"retriever({search_type}, {search_kwargs})"


class StubRetriever:
    """Returns a fixed ranking, so metric maths is checked without an index."""

    def __init__(self, ranking):
        self.ranking = ranking

    def invoke(self, query):
        return [Document(page_content="x", metadata={"source": s}) for s in self.ranking]


def q(sources=("right.md",)):
    return {"question": "anything", "relevant_sources": list(sources)}


# -- the bug: variants must not inherit ambient config --------------------------


def test_every_ablation_declares_its_own_lambda():
    """A spec that omits the lambda is how the original bug got in."""
    for name, spec in ABLATIONS.items():
        assert "mmr_lambda" in spec, f"{name} does not declare mmr_lambda"


def test_the_mmr_variants_actually_enable_mmr():
    for name in ("dense+mmr", "hybrid+mmr"):
        assert ABLATIONS[name]["mmr_lambda"] not in (None, 1.0), (
            f"{name} would be identical to its non-MMR twin"
        )


def test_the_non_mmr_variants_actually_disable_mmr():
    for name in ("dense-only", "hybrid"):
        assert ABLATIONS[name]["mmr_lambda"] is None


def test_mmr_and_non_mmr_variants_differ(monkeypatch):
    """dense-only and dense+mmr must not resolve to the same retriever."""
    monkeypatch.setattr(config, "MMR_LAMBDA", 1.0)  # the ambient value that hid the bug
    store = StubStore()
    retriever_for(store, [], ABLATIONS["dense+mmr"], 5)
    assert store.search_kwargs["lambda_mult"] == 0.5

    plain = retriever_for(StubStore(), [], ABLATIONS["dense-only"], 5)
    assert plain != store.search_kwargs


def test_a_variant_ignores_the_ambient_lambda(monkeypatch):
    monkeypatch.setattr(config, "MMR_LAMBDA", 0.1)
    store = StubStore()
    retriever_for(store, [], {"hybrid": False, "mmr_lambda": 0.8}, 5)
    assert store.search_kwargs["lambda_mult"] == 0.8


def test_building_a_variant_leaves_global_config_untouched(monkeypatch):
    """The old harness mutated config and restored it in a finally block."""
    before = (config.USE_HYBRID, config.MMR_LAMBDA, config.HYBRID_WEIGHTS)
    retriever_for(StubStore(), [], {"hybrid": False, "mmr_lambda": 0.5}, 5)
    assert (config.USE_HYBRID, config.MMR_LAMBDA, config.HYBRID_WEIGHTS) == before


def test_the_shipped_configs_no_longer_carry_the_misleading_mmr_flag():
    """CONFIGS['tuned'] used to say mmr: True while its label said MMR off."""
    for cfg in CONFIGS.values():
        assert "mmr" not in cfg
        assert "mmr_lambda" in cfg


def test_no_two_ablations_are_the_same_experiment():
    """hybrid+mmr(l=1.0) used to duplicate hybrid exactly."""
    seen = {}
    for name, spec in ABLATIONS.items():
        key = (spec["hybrid"], spec.get("mmr_lambda"), spec.get("weights"))
        assert key not in seen, f"{name} duplicates {seen.get(key)}"
        seen[key] = name


# -- retriever parameterisation ------------------------------------------------


def test_build_retriever_defaults_every_knob_to_config():
    store = StubStore()
    build_retriever(store, chunks=None, top_k=5)
    assert store.search_kwargs["lambda_mult"] == config.MMR_LAMBDA


def test_build_retriever_honours_explicit_overrides():
    store = StubStore()
    build_retriever(store, chunks=None, top_k=3, mmr_lambda=0.25, fetch_k=40)
    assert store.search_kwargs["k"] == 3
    assert store.search_kwargs["lambda_mult"] == 0.25
    assert store.search_kwargs["fetch_k"] == 40


def test_fetch_k_never_drops_below_four_times_k():
    """MMR needs a candidate pool meaningfully larger than the result set."""
    store = StubStore()
    build_retriever(store, chunks=None, top_k=10, fetch_k=5)
    assert store.search_kwargs["fetch_k"] == 40


def test_hybrid_is_skipped_when_there_are_no_chunks():
    """BM25 needs the raw documents; without them it must degrade, not crash."""
    store = StubStore()
    result = build_retriever(store, chunks=None, use_hybrid=True)
    assert isinstance(result, str)  # the stub dense retriever, not an ensemble


def test_dense_only_retriever_uses_plain_similarity():
    store = StubStore()
    dense_only_retriever(store, 7)
    assert store.search_type == "similarity"
    assert store.search_kwargs == {"k": 7}


# -- metric maths --------------------------------------------------------------


def test_a_relevant_chunk_at_rank_one_scores_perfectly():
    hit_rate, mrr, precision, _ = score(StubRetriever(["right.md"] * 5), [q()], 5)
    assert (hit_rate, mrr, precision) == (1.0, 1.0, 1.0)


def test_no_relevant_chunk_scores_zero():
    hit_rate, mrr, precision, _ = score(StubRetriever(["wrong.md"] * 5), [q()], 5)
    assert (hit_rate, mrr, precision) == (0.0, 0.0, 0.0)


def test_mrr_is_the_reciprocal_of_the_first_relevant_rank():
    ranking = ["wrong.md", "wrong.md", "right.md", "wrong.md", "wrong.md"]
    _, mrr, _, _ = score(StubRetriever(ranking), [q()], 5)
    assert mrr == pytest.approx(1 / 3)


def test_precision_counts_every_relevant_chunk_not_just_the_first():
    ranking = ["right.md", "right.md", "wrong.md", "wrong.md", "wrong.md"]
    _, _, precision, _ = score(StubRetriever(ranking), [q()], 5)
    assert precision == pytest.approx(0.4)


def test_hit_rate_averages_across_questions():
    retriever = StubRetriever(["right.md"] + ["wrong.md"] * 4)
    hit_rate, _, _, _ = score(retriever, [q(), q(["other.md"])], 5)
    assert hit_rate == 0.5


def test_only_the_top_k_are_scored():
    """A relevant chunk at rank 6 is not retrieved as far as the model is concerned."""
    ranking = ["wrong.md"] * 5 + ["right.md"]
    hit_rate, _, _, _ = score(StubRetriever(ranking), [q()], 5)
    assert hit_rate == 0.0


def test_the_detail_records_the_rank_and_the_tags():
    questions = [dict(q(), tags=["exact"])]
    _, _, _, detail = score(StubRetriever(["wrong.md", "right.md"]), questions, 5)
    assert detail[0]["first_relevant_rank"] == 2
    assert detail[0]["tags"] == ["exact"]


# -- per-class reporting -------------------------------------------------------


def test_by_tag_counts_hits_per_failure_class():
    detail = [
        {"tags": ["exact"], "hit": True},
        {"tags": ["exact"], "hit": False},
        {"tags": ["multihop"], "hit": True},
    ]
    assert by_tag(detail) == {"exact": "1/2", "multihop": "1/1"}


def test_untagged_questions_are_still_counted():
    assert by_tag([{"tags": [], "hit": True}]) == {"untagged": "1/1"}


# -- the baseline gate ---------------------------------------------------------


def result(name="tuned", **overrides):
    base = {"config": name, "label": "l", "k": 5, "questions": 36, "chunks": 1534,
            "hit_rate": 0.833, "mrr": 0.699, "precision": 0.467, "by_tag": {}}
    base.update(overrides)
    return base


@pytest.fixture
def baseline_file(tmp_path, monkeypatch):
    path = tmp_path / "baseline.json"

    def write(results):
        path.write_text(json.dumps({"corpus": {}, "results": results}), encoding="utf-8")

    monkeypatch.setattr("eval.evaluate.BASELINE_FILE", path)
    monkeypatch.setattr("eval.evaluate.CORPUS_LOCK", tmp_path / "absent.json")
    return write


def test_check_passes_an_unchanged_run(baseline_file):
    baseline_file([result()])
    assert check_against_baseline([result()], 0.02) == 0


def test_check_fails_a_regression_beyond_tolerance(baseline_file, capsys):
    baseline_file([result()])
    assert check_against_baseline([result(hit_rate=0.75)], 0.02) == 1
    assert "FAIL" in capsys.readouterr().out


def test_check_tolerates_a_small_dip(baseline_file):
    baseline_file([result()])
    assert check_against_baseline([result(hit_rate=0.82)], 0.02) == 0


def test_check_never_fails_on_an_improvement(baseline_file):
    baseline_file([result()])
    assert check_against_baseline([result(hit_rate=1.0, mrr=1.0, precision=1.0)], 0.02) == 0


def test_check_counts_every_regressed_metric(baseline_file):
    baseline_file([result()])
    regressed = result(hit_rate=0.5, mrr=0.2, precision=0.1)
    assert check_against_baseline([regressed], 0.02) == 3


def test_check_reports_a_config_missing_from_the_baseline(baseline_file, capsys):
    baseline_file([result("tuned")])
    check_against_baseline([result("brand-new")], 0.02)
    assert "NEW" in capsys.readouterr().out


def test_check_without_a_baseline_is_a_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("eval.evaluate.BASELINE_FILE", tmp_path / "nope.json")
    assert check_against_baseline([result()], 0.02) == 1
    assert "No baseline" in capsys.readouterr().out


# -- what gets checked in ------------------------------------------------------


def test_summarise_drops_the_per_question_dump():
    summary = summarise([dict(result(), detail=[{"question": "q"}] * 36)])
    assert "detail" not in summary["results"][0]
    assert all(m in summary["results"][0] for m in METRICS)


def test_summarise_records_what_it_was_measured_on():
    summary = summarise([result()])
    assert "corpus" in summary and "embedding_model" in summary


def test_the_checked_in_baseline_is_present_and_sane():
    """This file is the projects only defence against a silent regression."""
    from eval.evaluate import BASELINE_FILE

    assert BASELINE_FILE.is_file(), "no checked-in baseline"
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    names = {r["config"] for r in baseline["results"]}
    assert {"baseline", "tuned"} <= names
    assert baseline["corpus"].get("ref"), "baseline does not say which corpus it used"
    for r in baseline["results"]:
        for metric in METRICS:
            assert 0.0 <= r[metric] <= 1.0
