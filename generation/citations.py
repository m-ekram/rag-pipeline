"""Citation extraction and validation.

A grounded answer is only grounded if its citations point at evidence that was
actually supplied. Small local models fabricate reference numbers more readily
than large ones, so this is a hard check rather than a diagnostic: an answer
citing [7] when six chunks were supplied is a detected failure, not a warning.

This is also the second abstention signal the plan lists as Stretch ("LLM
self-check"): an answer with zero valid citations is ungrounded regardless of
how confident the retrieval score was.
"""

import re
from dataclasses import dataclass, field

from .prompts import ABSTAIN_TOKEN, BuiltPrompt

# Matches [1], [2,3], [1, 2] and [1][2]; tolerates whitespace.
_CITATION = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationReport:
    valid: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """Every citation resolves, and at least one exists."""
        return bool(self.valid) and not self.invalid

    @property
    def hallucinated_citation(self) -> bool:
        return bool(self.invalid)


def extract_citations(text: str) -> list[int]:
    """Return every cited number, in order of first appearance, deduplicated."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number not in seen:
                seen.append(number)
    return seen


def is_abstention(text: str) -> bool:
    """Did the model actually decline?

    Tolerant of the punctuation and casing small models wrap around a required
    literal, but NOT of instruction echo. A 7B model frequently restates its
    instructions before answering ("...reply exactly INSUFFICIENT_EVIDENCE. A
    Roth IRA is..."), and treating that as a refusal would record a correct,
    grounded answer as an abstention — inflating the abstention rate and
    corrupting the coverage axis of the risk-coverage curve.

    So the token counts only when it is the response, not when it is quoted
    inside one: it must appear on the first line, or be essentially the whole
    reply.
    """
    normalised = text.strip().strip(".\"\' ").upper()
    if not normalised:
        return False

    first_line = normalised.splitlines()[0]
    remainder = normalised.replace(ABSTAIN_TOKEN, " ").split()

    if ABSTAIN_TOKEN in first_line and len(first_line.replace(ABSTAIN_TOKEN, " ").split()) < 5:
        return True
    # Token buried further down, but nothing else of substance was said.
    return ABSTAIN_TOKEN in normalised and len(remainder) < 5


def validate_citations(answer: str, evidence_count: int,
                       *, check_uncited: bool = True, is_table: bool = False) -> CitationReport:
    """Check every citation resolves to supplied evidence."""
    report = CitationReport()
    for number in extract_citations(answer):
        if 1 <= number <= evidence_count:
            report.valid.append(number)
        elif is_table and evidence_count == 1 and 1 <= number <= 10:
            # Table Row Citation Relaxation: single table source with row numbers
            report.valid.append(number)
        else:
            report.invalid.append(number)

    if check_uncited:
        for sentence in _SENTENCE_END.split(answer.strip()):
            sentence = sentence.strip()
            # Short fragments are usually connective text, not factual claims.
            if len(sentence.split()) >= 5 and not _CITATION.search(sentence):
                report.uncited_sentences.append(sentence)

    return report


def render_citations(answer: str, built: BuiltPrompt) -> str:
    """Expand `[n]` markers into the plan's `[Document X, Section Y, Page Z]`.

    Unresolvable markers are left untouched so a hallucinated reference stays
    visible in the output rather than being silently swallowed.
    """
    is_table = False
    if len(built.evidence) == 1:
        first = built.evidence[0]
        meta = getattr(first, "metadata", {}) or {}
        first_text = getattr(first, "text", "") or ""
        is_table = meta.get("block_type") == "table" or "[Record:" in first_text or (first_text.count("|") >= 4)

    def replace(match: re.Match) -> str:
        rendered = []
        for part in match.group(1).split(","):
            num = int(part.strip())
            citation = built.citation_for(num)
            if citation:
                rendered.append(citation)
            elif is_table and len(built.evidence) == 1 and 1 <= num <= 10:
                base_cit = built.citation_for(1) or "[1]"
                rendered.append(f"{base_cit} (Row {num})")
            else:
                rendered.append(f"[{part.strip()}]")
        return " ".join(rendered)

    return _CITATION.sub(replace, answer)
