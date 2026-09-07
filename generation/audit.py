"""Post-Generation Fact Auditor for grounding verification.

Verifies that numerical facts, percentages, and IDs asserted in the LLM's answer
exist verbatim in the cited source evidence chunks, catching hallucinated figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Sequence

# Regex matching numbers, percentages, decimals, and structured IDs
_FACT_PATTERN = re.compile(
    r"\b(?:\d+(?:\.\d+)?%?|[A-Z]{2,4}/\d+/\d+/\d+|[A-Z]{3}\d{7})\b"
)


@dataclass
class FactAuditResult:
    """Outcome of verifying facts against source evidence."""

    verified: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True if no asserted numbers/IDs were fabricated."""
        return len(self.unverified) == 0


class FactAuditor:
    """Audits generated answers for verbatim grounding of numbers and entities."""

    def audit(self, answer: str, evidence_texts: Sequence[str]) -> FactAuditResult:
        """Extract asserted facts from answer and verify their presence in evidence."""
        combined_evidence = " ".join(evidence_texts)
        # Normalize whitespace
        combined_evidence = re.sub(r"\s+", " ", combined_evidence)

        # Extract candidates from answer
        asserted_facts = _FACT_PATTERN.findall(answer)
        verified: list[str] = []
        unverified: list[str] = []

        # Common citation markers [1], [2] to ignore
        citation_set = {f"[{i}]" for i in range(1, 20)}

        for fact in asserted_facts:
            # Skip single-digit citation numbers or ordinals
            if fact in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10") and f"[{fact}]" in answer:
                continue

            # Strip % for matching if raw number is in table
            fact_raw = fact.rstrip("%")
            if fact in combined_evidence or fact_raw in combined_evidence:
                if fact not in verified:
                    verified.append(fact)
            else:
                if fact not in unverified:
                    unverified.append(fact)

        return FactAuditResult(verified=verified, unverified=unverified)
