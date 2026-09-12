"""End-to-end grounded answering: retrieve -> rerank -> gate -> generate -> validate.

This is the object `/query` will wrap and the eval harness will drive. It
records what happened at every stage, because the plan's definition of done
requires per-stage latency and per-query token cost to be measured rather than
estimated.
"""

import time
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from generation.abstention import AbstentionResult, Decision, ThresholdGate
from generation.audit import FactAuditor, FactAuditResult
from generation.citations import (
    CitationReport,
    is_abstention,
    render_citations,
    validate_citations,
)
from generation.llm import LLMBackend, LLMResponse, get_llm
from generation.prompts import BuiltPrompt, build_prompt
from retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)

# Settings tuned for an 8B model on CPU, where prompt evaluation dominates.
# A 3000-token evidence budget is ~5 chunks of 200 words; on CPU that alone can
# cost 30-60s before a single output token appears. The grounded-answer prompt
# caps answers at 120 words, so 256 output tokens is already generous.
LOCAL_PRESET = {
    "evidence_limit": 3,
    "evidence_token_budget": 1200,
    "max_answer_tokens": 256,
}

# Router intents whose answer is a list of every matching record. The normal
# top-few evidence cap would silently truncate that list, so they get a larger
# cap and budget. Enum values rather than members, so this module does not
# depend on the router.
ROSTER_INTENTS = frozenset({"exhaustive_list", "relation_lookup", "house_lookup"})
# Fallback for retrievers that do not report an intent.
ROSTER_KEYWORDS = (
    "which voters", "list all", "who all", "who lives in", "find all",
    "voters in", "सभी", "किन",
)


@dataclass
class AnswerResult:
    question: str
    answer: str
    decision: Decision
    abstention: AbstentionResult
    evidence: list[ScoredChunk] = field(default_factory=list)
    # Evidence that actually reached the prompt. The budget can drop candidates,
    # so this is usually shorter than `evidence` — and it, not the candidate
    # count, is what citation numbers refer to.
    prompt_evidence: list = field(default_factory=list)
    evidence_dropped: int = 0
    citations: Optional[CitationReport] = None
    audit: Optional[FactAuditResult] = None
    llm: Optional[LLMResponse] = None
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.decision.is_abstention

    @property
    def grounded(self) -> bool:
        return self.citations is not None and self.citations.is_grounded

    def to_dict(self) -> dict[str, Any]:
        """Flat record for the experiment log (JSON/CSV + pandas)."""
        return {
            "question": self.question,
            "answer": self.answer,
            "decision": self.decision.value,
            "abstained": self.abstained,
            "gate_score": self.abstention.score,
            "gate_threshold": self.abstention.threshold,
            "grounded": self.grounded,
            "n_candidates": len(self.evidence),
            "n_evidence_used": len(self.prompt_evidence),
            "n_evidence_dropped": self.evidence_dropped,
            "hallucinated_citation": bool(
                self.citations and self.citations.hallucinated_citation
            ),
            "backend": self.llm.backend if self.llm else None,
            "model": self.llm.model if self.llm else None,
            "input_tokens": self.llm.input_tokens if self.llm else 0,
            "output_tokens": self.llm.output_tokens if self.llm else 0,
            "cost_usd": self.llm.cost_usd if self.llm else 0.0,
            **{f"latency_{k}_ms": round(v, 2) for k, v in self.latency_ms.items()},
        }


class RAGPipeline:
    def __init__(
        self,
        retriever,
        gate: ThresholdGate,
        *,
        reranker=None,
        llm: Optional[LLMBackend] = None,
        candidate_limit: int = 50,
        evidence_limit: int = 5,
        evidence_token_budget: int = 3000,
        max_answer_tokens: int = 512,
        roster_evidence_limit: int = 30,
        roster_token_budget: int = 3000,
    ):
        self.retriever = retriever
        self.gate = gate
        self.reranker = reranker
        self._llm = llm
        self.candidate_limit = candidate_limit
        self.evidence_limit = evidence_limit
        self.evidence_token_budget = evidence_token_budget
        self.max_answer_tokens = max_answer_tokens
        # List-style questions ("which voters...") need every matching record
        # in the prompt; the limits above are sized for a single best answer.
        self.roster_evidence_limit = roster_evidence_limit
        self.roster_token_budget = roster_token_budget

    @classmethod
    def for_local_model(cls, retriever, gate: ThresholdGate, **kwargs):
        """Build a pipeline with CPU-friendly prompt sizes.

        Explicit keyword arguments still win, so the preset is a default, not a
        ceiling.
        """
        return cls(retriever, gate, **{**LOCAL_PRESET, **kwargs})

    def warmup(self) -> float:
        """Load the model before the first question rather than during it.

        On a cold Ollama the first request pays a multi-GB model load; doing it
        here keeps that cost out of the user's first query and out of the
        request timeout.
        """
        warm = getattr(self.llm, "warmup", None)
        if warm is None:
            return 0.0
        seconds = warm()
        logger.info("Warmed %s in %.1fs", getattr(self.llm, "model", "?"), seconds)
        return seconds

    @property
    def llm(self) -> LLMBackend:
        # Resolved lazily so an abstain-only run needs no backend at all.
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _decompose_query(self, question: str) -> list[str]:
        """Decompose multi-hop or composite questions into sub-queries."""
        # Detect multiple table references (e.g. Table 26 ... Table 28)
        table_matches = re.findall(r"(?:Table|Tab\.?|सारणी)\s*\d+", question, re.IGNORECASE)
        if len(table_matches) >= 2:
            return [f"{tm} in {question}" for tm in table_matches]

        # Split on explicit conjunctions if the question compares two aspects
        conjunctions = [r"\s+(?:AND|and|तथा|और|versus|vs\.?)\s+"]
        for conj in conjunctions:
            parts = re.split(conj, question)
            if len(parts) == 2 and len(parts[0].strip()) > 15 and len(parts[1].strip()) > 15:
                return [parts[0].strip(), parts[1].strip()]
        return [question]

    def _is_roster_query(self, question: str, intents) -> bool:
        """Does the question ask for every matching record rather than the best one?

        The retriever's reported intent is authoritative; the keyword list only
        covers retrievers that report none.
        """
        known = [getattr(i, "value", i) for i in intents if i is not None]
        if known:
            return any(i in ROSTER_INTENTS for i in known)
        lowered = question.lower()
        return any(kw in lowered for kw in ROSTER_KEYWORDS)

    def answer(
        self,
        question: str,
        *,
        stream_callback: Optional[Any] = None,
        target_lang: Optional[str] = None,
        on_stage: Optional[Any] = None,
    ) -> AnswerResult:
        """Answer `question`; `on_stage(name, message)` is told as each stage starts."""
        timings: dict[str, float] = {}
        stage = on_stage or (lambda name, message: None)

        stage("retrieval", "Searching and ranking evidence...")
        started = time.perf_counter()
        sub_queries = self._decompose_query(question)
        intents = []
        if len(sub_queries) > 1:
            all_candidates = []
            seen_ids = set()
            for sq in sub_queries:
                for c in self.retriever.retrieve(sq, limit=self.candidate_limit):
                    cid = getattr(c, "chunk_id", None) or getattr(c.chunk, "chunk_id", str(c))
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        all_candidates.append(c)
                intents.append(getattr(self.retriever, "last_intent", None))
            candidates = all_candidates
        else:
            candidates = self.retriever.retrieve(question, limit=self.candidate_limit)
            intents.append(getattr(self.retriever, "last_intent", None))
        timings["retrieval"] = (time.perf_counter() - started) * 1000

        # Decided before any truncation: slicing to the single-answer limit
        # first is what capped list answers at 5 records.
        roster = self._is_roster_query(question, intents)
        evidence_limit = (
            max(self.evidence_limit, self.roster_evidence_limit) if roster else self.evidence_limit
        )

        # Only rerank if candidates were not already reranked or are not exact structural hits (score >= 0.95)
        has_exact_hits = any(getattr(c, "score", 0.0) >= 0.95 for c in candidates)
        retriever_reranks = getattr(self.retriever, "reranker", None) is not None

        if self.reranker is not None and candidates and not has_exact_hits and not retriever_reranks:
            stage("rerank", f"Reranking {len(candidates)} candidates...")
            started = time.perf_counter()
            candidates = self.reranker.rerank(
                question, candidates, limit=evidence_limit
            )
            timings["rerank"] = (time.perf_counter() - started) * 1000
        else:
            candidates = candidates[:evidence_limit]

        gate_result = self.gate.decide(candidates)

        # Abstaining before generation is the point: it is the correct answer
        # *and* it skips the slowest stage, which dominates on a local model.
        if gate_result.decision.is_abstention:
            return AnswerResult(
                question=question,
                answer="",
                decision=gate_result.decision,
                abstention=gate_result,
                evidence=list(candidates),
                latency_ms=timings,
            )

        built: BuiltPrompt = build_prompt(
            question,
            candidates,
            evidence_token_budget=(
                max(self.evidence_token_budget, self.roster_token_budget)
                if roster else self.evidence_token_budget
            ),
            max_evidence=evidence_limit,
            target_lang=target_lang,
        )

        complete_kwargs: dict[str, Any] = {
            "system": built.system,
            "max_tokens": self.max_answer_tokens,
        }
        if stream_callback is not None:
            import inspect
            sig = inspect.signature(self.llm.complete)
            if "stream_callback" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                complete_kwargs["stream_callback"] = stream_callback

        stage("generation", f"Generating with {getattr(self.llm, 'name', 'model')} · "
                            f"{getattr(self.llm, 'model', '')}...")
        started = time.perf_counter()
        response = self.llm.complete(
            built.prompt,
            **complete_kwargs
        )
        timings["generation"] = (time.perf_counter() - started) * 1000

        # The model can abstain even when the gate let it through — that is a
        # genuine second signal, not an error, and the plan counts it as one.
        if is_abstention(response.text):
            return AnswerResult(
                question=question,
                answer="",
                decision=Decision.ABSTAIN_IRRELEVANT,
                abstention=AbstentionResult(
                    decision=Decision.ABSTAIN_IRRELEVANT,
                    score=gate_result.score,
                    threshold=gate_result.threshold,
                    reason="model declined: insufficient evidence",
                ),
                evidence=list(candidates),
                prompt_evidence=list(built.evidence),
                evidence_dropped=built.dropped,
                llm=response,
                latency_ms=timings,
            )

        is_table = False
        if len(built.evidence) == 1:
            first = built.evidence[0]
            meta = getattr(first, "metadata", {}) or {}
            first_text = getattr(first, "text", "") or ""
            is_table = meta.get("block_type") == "table" or "[Record:" in first_text or (first_text.count("|") >= 4)

        report = validate_citations(response.text, len(built.evidence), is_table=is_table)
        if report.hallucinated_citation:
            logger.warning("Answer cited non-existent evidence: %s", report.invalid)

        rendered_answer = render_citations(response.text, built)
        # Audit raw response text so expanded document IDs/titles (e.g. 'pmp-2031-report') do not trigger false alerts
        audit_result = FactAuditor().audit(
            response.text, [c.text for c in built.evidence]
        )
        if not audit_result.is_clean:
            logger.warning("FactAuditor detected unverified facts: %s", audit_result.unverified)

        return AnswerResult(
            question=question,
            answer=rendered_answer,
            decision=Decision.ANSWER,
            abstention=gate_result,
            evidence=list(candidates),
            prompt_evidence=list(built.evidence),
            evidence_dropped=built.dropped,
            citations=report,
            audit=audit_result,
            llm=response,
            latency_ms=timings,
        )
