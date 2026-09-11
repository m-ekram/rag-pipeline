"""Grounded-generation prompt construction and evidence packing.

Two design choices here are driven by having to work on a small local model,
not just on Claude:

1. **Numbered citations, not prose citations.** The plan's output format is
   `[Document X, Section Y, Page Z]`, but asking a 7B model to reproduce a
   three-field citation verbatim invites malformed and hallucinated references.
   The model is instead asked for `[1]`, `[2]` against numbered evidence, and
   the full plan-format citation is rendered deterministically afterwards. The
   user-visible format is unchanged; only the model's job got easier, and
   validation becomes exact rather than fuzzy.

2. **An explicit evidence budget.** A local model's practical context is far
   smaller than its advertised one, and prompt length dominates latency on CPU.
   Evidence is packed to a token budget and truncated at chunk boundaries so a
   citation never points at text the model could not see.
"""

from dataclasses import dataclass
import re
from typing import Iterable, Optional, Sequence

from ingestion.documents import Chunk
from retrieval.types import ScoredChunk

URDU_PATTERN = re.compile(r"[\u0600-\u06FF]")
HINDI_PATTERN = re.compile(r"[\u0900-\u097F]")

# Regex for detecting when the user explicitly requests an output language
_LANG_EN_PATTERN = re.compile(
    r"(?:(?:answer|reply|respond|give\s+answer)\s+in\s+english|\bin\s+english\b|انگریزی\s*میں|انگلش\s*میں|अंग्रेज़ी?\s*में|इंग्लिश\s*में)",
    re.IGNORECASE,
)
_LANG_HI_PATTERN = re.compile(
    r"(?:(?:answer|reply|respond|give\s+answer)\s+in\s+hindi|\bin\s+hindi\b|ہندی\s*میں|हिन्द?ी\s*में)",
    re.IGNORECASE,
)
_LANG_UR_PATTERN = re.compile(
    r"(?:(?:answer|reply|respond|give\s+answer)\s+in\s+urdu|\bin\s+urdu\b|اردو\s*میں|उर्दू\s*में)",
    re.IGNORECASE,
)

# The abstention instruction is deliberately blunt and repeated: small models
# comply far more reliably with an explicit refusal token than with a nuanced
# "if you are unsure" hedge.
SYSTEM_PROMPT = """\
You answer questions using ONLY the numbered evidence provided.

CRITICAL CITATION RULES:
1. Cite evidence using ONLY the source indices [1], [2], etc., provided in the numbered evidence list below.
2. NEVER output original paper reference numbers (such as [64], [66], [18], [21]) as citations.
3. Every factual claim, number, and statistic must cite its source index [1], [2].
4. If information is missing from the provided sources, state exactly: INSUFFICIENT_EVIDENCE.
5. Answer in the language of the question. If the evidence is in another language (such as Hindi/Devanagari), comprehend and extract the facts from the evidence accurately into your answer.
6. Do not guess. Do not speculate. Do not apologise or explain your reasoning.
7. Keep the answer under 120 words.
8. In electoral roll records formatted as - [Serial: ... | EPIC: ... | Voter: ... | Relation: ... | House: ... | Age: ... | Gender: ...], each line corresponds strictly to one person. NEVER mix or combine fields from different voter lines.
9. When asked for details of an entity (such as ID, name, code, or serial), extract and present all available fields from that record in the evidence. Do not decline with INSUFFICIENT_EVIDENCE if the record appears in the sources.
10. When asked to list or find multiple entities (such as "Which voters...", "Who lives in...", "List all..."), list ALL matching records found across the provided sources by name and serial number with citation [n]. If any matching records exist in the evidence, you MUST answer and NEVER output INSUFFICIENT_EVIDENCE."""

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

PROMPT_TEMPLATE = """\
Evidence:
{evidence}

Question: {question}

Answer using only the evidence above, citing sources as [n]. \
If some sources match the question, answer using the matching sources and ignore unrelated ones. \
Only if NO evidence answers the question, reply exactly {abstain}."""

# English averages ~1.3 tokens per whitespace word across tokenisers. This is an
# estimate used only for *packing*; every reported token count comes from the
# backend's own usage figures, never from here.
TOKENS_PER_WORD = 1.33


def estimate_tokens(text: str) -> int:
    """Cheap, backend-agnostic token estimate for budgeting."""
    return int(len(text.split()) * TOKENS_PER_WORD) + 1


@dataclass
class BuiltPrompt:
    """A prompt plus the evidence that actually fitted into it."""

    prompt: str
    system: str
    evidence: list[Chunk]          # index i here is citation [i+1]
    dropped: int = 0               # candidates that did not fit the budget
    estimated_tokens: int = 0

    def citation_for(self, number: int) -> Optional[str]:
        """Render `[n]` back into the plan's `[Document X, Section Y, Page Z]`."""
        if 1 <= number <= len(self.evidence):
            return self.evidence[number - 1].citation()
        return None


def _as_chunks(candidates: Iterable) -> list[Chunk]:
    chunks = []
    seen_parents = set()
    for item in candidates:
        chunk = item.chunk if isinstance(item, ScoredChunk) else item
        if chunk is not None:
            parent_id = (chunk.metadata or {}).get("parent_id")
            parent_text = (chunk.metadata or {}).get("parent_text")

            # Avoid duplicating identical parent tables/sections if multiple child rows matched
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)

            if parent_text and parent_text != chunk.text:
                chunk_view = Chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=parent_text,
                    ordinal=chunk.ordinal,
                    title=chunk.title,
                    source=chunk.source,
                    page=chunk.page,
                    section=chunk.section,
                    metadata=chunk.metadata,
                )
                chunks.append(chunk_view)
            else:
                chunks.append(chunk)
    return chunks


def format_evidence(chunks: Sequence[Chunk]) -> str:
    """Number the evidence so the model can cite it positionally."""
    return "\n\n".join(
        f"[{i}] {chunk.text.strip()}" for i, chunk in enumerate(chunks, 1)
    )


def build_prompt(
    question: str,
    candidates: Iterable,
    *,
    evidence_token_budget: int = 3000,
    max_evidence: int = 8,
    system: str = SYSTEM_PROMPT,
    target_lang: Optional[str] = None,
) -> BuiltPrompt:
    """Pack the highest-ranked evidence that fits, in rank order.

    Chunks are included whole or not at all: a half-included chunk would let the
    model cite text it never received.
    """
    chunks = _as_chunks(candidates)[:max_evidence]

    kept: list[Chunk] = []
    used = 0
    dropped = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk.text) + 4  # marker + separator
        if kept and used + cost > evidence_token_budget:
            dropped += 1
            continue
        # Always keep the top chunk even if it alone blows the budget; returning
        # zero evidence would turn a retrieval result into a false abstention.
        kept.append(chunk)
        used += cost

    resolved_lang = None
    if target_lang and target_lang.lower() not in ("auto", "none"):
        t = target_lang.lower()
        if t in ("en", "english"):
            resolved_lang = "en"
        elif t in ("hi", "hindi"):
            resolved_lang = "hi"
        elif t in ("ur", "urdu"):
            resolved_lang = "ur"
        else:
            resolved_lang = t

    if not resolved_lang:
        # Check if user explicitly asked for a language inside the question text
        if _LANG_EN_PATTERN.search(question):
            resolved_lang = "en"
        elif _LANG_HI_PATTERN.search(question):
            resolved_lang = "hi"
        elif _LANG_UR_PATTERN.search(question):
            resolved_lang = "ur"
        # Otherwise default to the script of the question
        elif URDU_PATTERN.search(question):
            resolved_lang = "ur"
        elif HINDI_PATTERN.search(question):
            resolved_lang = "hi"

    lang_instruction = ""
    if resolved_lang == "ur":
        lang_instruction = "\nAnswer strictly in Urdu language (جواب لازمی طور پر اردو زبان میں دیں)."
        if system == SYSTEM_PROMPT:
            system = f"{system}\n9. You MUST write your final answer in Urdu language (جواب لازمی طور پر اردو زبان میں دیں)."
    elif resolved_lang == "hi":
        lang_instruction = "\nAnswer strictly in Hindi language (उत्तर हिन्दी भाषा में दें)."
        if system == SYSTEM_PROMPT:
            system = f"{system}\n9. You MUST write your final answer in Hindi language (उत्तर हिन्दी भाषा में दें)."
    elif resolved_lang == "en":
        # If question was in another script or explicitly requested English
        if URDU_PATTERN.search(question) or HINDI_PATTERN.search(question) or _LANG_EN_PATTERN.search(question) or (target_lang and target_lang.lower() in ("en", "english")):
            lang_instruction = "\nAnswer strictly in English language."
            if system == SYSTEM_PROMPT:
                system = f"{system}\n9. You MUST write your final answer strictly in English language."

    prompt = PROMPT_TEMPLATE.format(
        evidence=format_evidence(kept) if kept else "(none)",
        question=question.strip(),
        abstain=ABSTAIN_TOKEN,
    )
    if lang_instruction:
        prompt = f"{prompt}{lang_instruction}"

    return BuiltPrompt(
        prompt=prompt,
        system=system,
        evidence=kept,
        dropped=dropped,
        estimated_tokens=estimate_tokens(prompt) + estimate_tokens(system),
    )
