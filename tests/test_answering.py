"""Offline tests for prompts, citations, abstention, reranking and the pipeline.

All of this is prerequisite to testing against a real local model: it is the
logic a model's output is judged by, so it must be correct before any weights
are downloaded. Nothing here needs a model, a server or a key.
"""

import pytest

from generation.abstention import Decision, ThresholdGate
from generation.citations import (
    extract_citations,
    is_abstention,
    render_citations,
    validate_citations,
)
from generation.llm import LLMResponse
from generation.pipeline import RAGPipeline
from generation.prompts import ABSTAIN_TOKEN, build_prompt, estimate_tokens, format_evidence
from ingestion.documents import Chunk
from rerank.cross_encoder import CrossEncoderReranker, _sigmoid
from retrieval.types import ScoredChunk


def _chunk(cid, text, **kw):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=text, ordinal=0, **kw)


def _scored(cid, text, score, rank=1, **kw):
    return ScoredChunk(chunk_id=cid, score=score, rank=rank, chunk=_chunk(cid, text, **kw))


CANDIDATES = [
    _scored("a::0", "A Roth IRA is funded with post-tax dollars.", 0.91, 1),
    _scored("b::0", "A traditional IRA is funded pre-tax.", 0.80, 2),
    _scored("c::0", "Mortgage interest may be deductible.", 0.10, 3),
]


# --- prompt construction ------------------------------------------------


def test_evidence_is_numbered_for_positional_citation():
    text = format_evidence([c.chunk for c in CANDIDATES])
    assert text.startswith("[1] A Roth IRA")
    assert "[2] A traditional IRA" in text


def test_build_prompt_includes_question_and_abstain_instruction():
    built = build_prompt("What is a Roth IRA?", CANDIDATES)
    assert "What is a Roth IRA?" in built.prompt
    assert ABSTAIN_TOKEN in built.prompt
    assert len(built.evidence) == 3


def test_build_prompt_detects_urdu_and_hindi_script():
    urdu_q = "اس پولنگ بوتھ کا نام کیا ہے؟"
    built_urdu = build_prompt(urdu_q, CANDIDATES)
    assert "اردو زبان میں دیں" in built_urdu.prompt
    assert "اردو زبان میں دیں" in built_urdu.system

    hindi_q = "मतदान केंद्र का नाम क्या है?"
    built_hindi = build_prompt(hindi_q, CANDIDATES)
    assert "हिन्दी भाषा में दें" in built_hindi.prompt
    assert "हिन्दी भाषा में दें" in built_hindi.system



def test_build_prompt_respects_the_token_budget():
    big = [_scored(f"{i}::0", " ".join(["word"] * 400), 0.9, i) for i in range(5)]
    built = build_prompt("q", big, evidence_token_budget=600)
    assert len(built.evidence) < 5
    assert built.dropped > 0


def test_build_prompt_always_keeps_the_top_chunk():
    """Dropping everything would turn a real retrieval hit into a false abstention."""
    huge = [_scored("a::0", " ".join(["word"] * 5000), 0.9, 1)]
    built = build_prompt("q", huge, evidence_token_budget=10)
    assert len(built.evidence) == 1


def test_build_prompt_never_includes_partial_chunks():
    big = [_scored(f"{i}::0", " ".join(["word"] * 300), 0.9, i) for i in range(4)]
    built = build_prompt("q", big, evidence_token_budget=800)
    for kept in built.evidence:
        assert len(kept.text.split()) == 300  # whole or not at all


def test_build_prompt_caps_evidence_count():
    many = [_scored(f"{i}::0", "short text", 0.9, i) for i in range(20)]
    assert len(build_prompt("q", many, max_evidence=4).evidence) == 4


def test_build_prompt_handles_no_candidates():
    built = build_prompt("q", [])
    assert built.evidence == []
    assert "(none)" in built.prompt


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("one two three") > estimate_tokens("one")


def test_citation_for_maps_number_back_to_plan_format():
    built = build_prompt("q", [_scored("a::0", "text", 0.9, 1,
                                       title="Annual Report", page=7)])
    assert built.citation_for(1) == "[Document Annual Report, Page 7]"
    assert built.citation_for(2) is None


# --- citation extraction / validation -----------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Fact [1].", [1]),
    ("Fact [1] and [2].", [1, 2]),
    ("Fact [1, 2].", [1, 2]),
    ("Fact [1,2] and [1].", [1, 2]),
    ("Fact [ 3 ].", [3]),
    ("No citation here.", []),
])
def test_extract_citations(text, expected):
    assert extract_citations(text) == expected


def test_validate_flags_citations_beyond_supplied_evidence():
    """A 7B model citing [7] when 3 chunks were supplied is a detected failure."""
    report = validate_citations("Claim [1]. Other claim [7].", evidence_count=3)
    assert report.valid == [1]
    assert report.invalid == [7]
    assert report.hallucinated_citation is True
    assert report.is_grounded is False


def test_validate_accepts_a_fully_cited_answer():
    report = validate_citations("A Roth IRA is post-tax [1].", evidence_count=2)
    assert report.is_grounded is True
    assert report.uncited_sentences == []


def test_validate_reports_uncited_claims():
    report = validate_citations(
        "A Roth IRA is post-tax [1]. Interest rates will certainly rise next year.",
        evidence_count=1,
    )
    assert report.uncited_sentences == ["Interest rates will certainly rise next year."]
    # An uncited claim does not by itself make the answer ungrounded...
    assert report.is_grounded is True


def test_answer_with_no_citations_at_all_is_not_grounded():
    assert validate_citations("Just prose.", evidence_count=3).is_grounded is False


@pytest.mark.parametrize("text", [
    ABSTAIN_TOKEN,
    f"{ABSTAIN_TOKEN}.",
    f"  {ABSTAIN_TOKEN}  ",
    f'"{ABSTAIN_TOKEN}"',
    f"insufficient_evidence",
])
def test_is_abstention_tolerates_model_formatting_noise(text):
    assert is_abstention(text) is True


def test_is_abstention_false_for_a_real_answer():
    assert is_abstention("A Roth IRA is funded post-tax [1].") is False


def test_render_expands_numbers_into_plan_citation_format():
    built = build_prompt("q", [_scored("a::0", "t", 0.9, 1, title="Doc A", page=2)])
    assert render_citations("Claim [1].", built) == "Claim [Document Doc A, Page 2]."


def test_render_leaves_hallucinated_markers_visible():
    """Silently swallowing a bad reference would hide the failure."""
    built = build_prompt("q", [_scored("a::0", "t", 0.9, 1)])
    assert "[9]" in render_citations("Claim [9].", built)


# --- abstention gate ----------------------------------------------------


def test_gate_answers_above_threshold():
    result = ThresholdGate(0.5).decide(CANDIDATES)
    assert result.decision is Decision.ANSWER
    assert result.answered is True


def test_gate_abstains_as_irrelevant_below_threshold():
    weak = [_scored("c::0", "unrelated", 0.10, 1)]
    result = ThresholdGate(0.5).decide(weak)
    assert result.decision is Decision.ABSTAIN_IRRELEVANT
    assert "below threshold" in result.reason


def test_gate_distinguishes_no_evidence_from_irrelevant_evidence():
    """The plan's taxonomy needs these separable, not collapsed into 'didn't answer'."""
    empty = ThresholdGate(0.5).decide([])
    assert empty.decision is Decision.ABSTAIN_NO_EVIDENCE
    assert empty.score is None
    assert empty.decision is not Decision.ABSTAIN_IRRELEVANT


def test_gate_uses_top_score_only():
    mixed = [_scored("a::0", "x", 0.9, 1), _scored("b::0", "y", 0.01, 2)]
    assert ThresholdGate(0.5).decide(mixed).decision is Decision.ANSWER


def test_decision_is_abstention_property():
    assert Decision.ANSWER.is_abstention is False
    assert Decision.ABSTAIN_IRRELEVANT.is_abstention is True
    assert Decision.ABSTAIN_NO_EVIDENCE.is_abstention is True


# --- reranker -----------------------------------------------------------


class _FakeCrossEncoder:
    """Returns raw logits like ms-marco-MiniLM does, in reverse input order."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, batch_size=32):
        self.calls.append(pairs)
        return self.scores[: len(pairs)]


def test_reranker_reorders_by_score_and_normalises_logits():
    fake = _FakeCrossEncoder([-5.0, 8.0, 1.0])
    reranked = CrossEncoderReranker(model=fake).rerank("q", CANDIDATES)
    assert [r.chunk_id for r in reranked] == ["b::0", "c::0", "a::0"]
    assert all(0.0 < r.score < 1.0 for r in reranked)
    assert reranked[0].score == pytest.approx(_sigmoid(8.0))
    assert [r.rank for r in reranked] == [1, 2, 3]


def test_reranker_can_return_raw_logits():
    fake = _FakeCrossEncoder([-5.0, 8.0, 1.0])
    reranked = CrossEncoderReranker(model=fake, normalise=False).rerank("q", CANDIDATES)
    assert reranked[0].score == 8.0


def test_reranker_respects_limit():
    fake = _FakeCrossEncoder([1.0, 2.0, 3.0])
    assert len(CrossEncoderReranker(model=fake).rerank("q", CANDIDATES, limit=2)) == 2


def test_reranker_drops_candidates_without_payload():
    fake = _FakeCrossEncoder([1.0])
    bare = [ScoredChunk("x::0", 0.5)]
    assert CrossEncoderReranker(model=fake).rerank("q", bare) == []


def test_sigmoid_is_stable_at_extremes():
    """The naive 1/(1+exp(-x)) raises OverflowError here; the branched form
    underflows to 0.0 instead. Real logits are ~[-11, 11] — this is robustness."""
    assert _sigmoid(-800.0) == 0.0
    assert _sigmoid(800.0) == pytest.approx(1.0)
    assert _sigmoid(0.0) == 0.5
    assert all(0.0 <= _sigmoid(x) <= 1.0 for x in (-800, -11, 0, 11, 800))


# --- pipeline -----------------------------------------------------------


class _FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, limit=10):
        return self.results[:limit]


class _FakeLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, text):
        self.text = text
        self.prompts = []

    def complete(self, prompt, *, system=None, max_tokens=1024, temperature=0.0):
        self.prompts.append((system, prompt))
        return LLMResponse(text=self.text, model=self.model, backend=self.name,
                           input_tokens=100, output_tokens=20)

    def available(self):
        return True


def test_pipeline_answers_and_renders_citations():
    llm = _FakeLLM("A Roth IRA is post-tax [1].")
    pipe = RAGPipeline(_FakeRetriever(CANDIDATES), ThresholdGate(0.5), llm=llm)
    result = pipe.answer("What is a Roth IRA?")

    assert result.decision is Decision.ANSWER
    assert result.grounded is True
    assert "[Document a]" in result.answer
    assert result.llm.input_tokens == 100


def test_pipeline_abstains_without_calling_the_llm():
    """The gate must short-circuit the slowest stage, not just label the output."""
    llm = _FakeLLM("should never run")
    weak = [_scored("c::0", "unrelated", 0.1, 1)]
    pipe = RAGPipeline(_FakeRetriever(weak), ThresholdGate(0.5), llm=llm)
    result = pipe.answer("sourdough starter?")

    assert result.decision is Decision.ABSTAIN_IRRELEVANT
    assert result.answer == ""
    assert llm.prompts == []
    assert "generation" not in result.latency_ms


def test_pipeline_reports_no_evidence_separately():
    pipe = RAGPipeline(_FakeRetriever([]), ThresholdGate(0.5), llm=_FakeLLM("x"))
    assert pipe.answer("q").decision is Decision.ABSTAIN_NO_EVIDENCE


def test_pipeline_honours_a_model_side_abstention():
    llm = _FakeLLM(ABSTAIN_TOKEN)
    pipe = RAGPipeline(_FakeRetriever(CANDIDATES), ThresholdGate(0.5), llm=llm)
    result = pipe.answer("q")
    assert result.decision is Decision.ABSTAIN_IRRELEVANT
    assert "model declined" in result.abstention.reason
    assert result.llm is not None  # the call happened and is still costed


def test_pipeline_flags_hallucinated_citations():
    llm = _FakeLLM("Claim [1]. Another [9].")
    pipe = RAGPipeline(_FakeRetriever(CANDIDATES), ThresholdGate(0.5), llm=llm)
    result = pipe.answer("q")
    assert result.citations.hallucinated_citation is True
    assert result.grounded is False
    assert result.to_dict()["hallucinated_citation"] is True


def test_pipeline_runs_the_reranker_and_gates_on_its_score():
    """Gate thresholds the *reranked* score, not the retriever's."""
    fake = _FakeCrossEncoder([-9.0, -9.0, -9.0])  # sigmoid(-9) ~ 0.0001
    pipe = RAGPipeline(
        _FakeRetriever(CANDIDATES),
        ThresholdGate(0.5),
        reranker=CrossEncoderReranker(model=fake),
        llm=_FakeLLM("x"),
    )
    result = pipe.answer("q")
    assert result.decision is Decision.ABSTAIN_IRRELEVANT
    assert "rerank" in result.latency_ms


def test_pipeline_record_is_flat_and_loggable():
    llm = _FakeLLM("Claim [1].")
    pipe = RAGPipeline(_FakeRetriever(CANDIDATES), ThresholdGate(0.5), llm=llm)
    record = pipe.answer("q").to_dict()

    for key in ("decision", "abstained", "gate_score", "grounded",
                "input_tokens", "cost_usd", "latency_retrieval_ms"):
        assert key in record
    assert all(not isinstance(v, (list, dict)) for v in record.values())


def test_pipeline_needs_no_llm_when_it_abstains():
    """An abstain-only sweep must not require a backend to exist at all."""
    pipe = RAGPipeline(_FakeRetriever([]), ThresholdGate(0.5))
    assert pipe.answer("q").decision is Decision.ABSTAIN_NO_EVIDENCE


# --- local-model ergonomics ---------------------------------------------


def test_local_preset_shrinks_the_prompt():
    """Prompt evaluation dominates on CPU, so the local preset trims evidence."""
    from generation.pipeline import LOCAL_PRESET

    pipe = RAGPipeline.for_local_model(_FakeRetriever(CANDIDATES), ThresholdGate(0.5),
                                       llm=_FakeLLM("Claim [1]."))
    assert pipe.evidence_limit == LOCAL_PRESET["evidence_limit"] == 3
    assert pipe.evidence_token_budget == 1200
    assert pipe.max_answer_tokens == 256


def test_explicit_arguments_beat_the_local_preset():
    pipe = RAGPipeline.for_local_model(_FakeRetriever(CANDIDATES), ThresholdGate(0.5),
                                       llm=_FakeLLM("x"), evidence_limit=5)
    assert pipe.evidence_limit == 5


def test_warmup_delegates_to_the_backend():
    class _Warmable(_FakeLLM):
        def __init__(self):
            super().__init__("x")
            self.warmed = False

        def warmup(self):
            self.warmed = True
            return 1.5

    llm = _Warmable()
    pipe = RAGPipeline(_FakeRetriever(CANDIDATES), ThresholdGate(0.5), llm=llm)
    assert pipe.warmup() == 1.5
    assert llm.warmed is True


def test_warmup_is_a_no_op_for_backends_without_one():
    pipe = RAGPipeline(_FakeRetriever(CANDIDATES), ThresholdGate(0.5),
                       llm=_FakeLLM("x"))
    assert pipe.warmup() == 0.0
