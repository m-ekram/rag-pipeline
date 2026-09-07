"""The abstention gate — the project's central mechanism.

Phase 2 ships v1: a single threshold on the top reranked score. Phase 3
calibrates it on a held-out split and reports the full risk-coverage curve.

Three outcomes, matching the plan's taxonomy, because "didn't answer" hides the
distinction that matters:

  ANSWER               evidence looks sufficient
  ABSTAIN_IRRELEVANT   evidence was retrieved but scores below threshold
  ABSTAIN_NO_EVIDENCE  retrieval returned nothing at all

The gate runs *before* generation on purpose. On a local model the LLM call is
by far the slowest stage, so abstaining early is both the correct behaviour and
the single biggest throughput win in a sweep.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from retrieval.types import ScoredChunk


class Decision(str, Enum):
    ANSWER = "answer"
    ABSTAIN_IRRELEVANT = "abstain_irrelevant"
    ABSTAIN_NO_EVIDENCE = "abstain_no_evidence"

    @property
    def is_abstention(self) -> bool:
        return self is not Decision.ANSWER


@dataclass
class AbstentionResult:
    decision: Decision
    score: Optional[float]
    threshold: float
    reason: str

    @property
    def answered(self) -> bool:
        return self.decision is Decision.ANSWER


class ThresholdGate:
    """Single-threshold gate on the top-ranked score (abstention v1).

    `threshold` is a placeholder until Phase 3 calibration — it is deliberately
    a constructor argument with no clever default, so an uncalibrated number can
    never masquerade as a tuned one in the results.
    """

    def __init__(self, threshold: float, *, min_candidates: int = 1):
        self.threshold = threshold
        self.min_candidates = min_candidates

    def decide(self, candidates: Sequence[ScoredChunk]) -> AbstentionResult:
        if not candidates or len(candidates) < self.min_candidates:
            reason = (
                "retrieval returned no candidates" if not candidates
                else f"only {len(candidates)} candidate(s), "
                     f"minimum is {self.min_candidates}"
            )
            return AbstentionResult(
                decision=Decision.ABSTAIN_NO_EVIDENCE,
                score=candidates[0].score if candidates else None,
                threshold=self.threshold,
                reason=reason,
            )

        top = candidates[0].score
        if top < self.threshold:
            return AbstentionResult(
                decision=Decision.ABSTAIN_IRRELEVANT,
                score=top,
                threshold=self.threshold,
                reason=f"top score {top:.4f} below threshold {self.threshold:.4f}",
            )

        return AbstentionResult(
            decision=Decision.ANSWER,
            score=top,
            threshold=self.threshold,
            reason=f"top score {top:.4f} meets threshold {self.threshold:.4f}",
        )
