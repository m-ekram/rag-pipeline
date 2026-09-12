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
# Kept short on purpose: on a CPU the model reads the prompt at ~30-40 tokens
# per second, so every rule costs latency on every question. The previous
# ten-rule prompt (~440 tokens, mostly about voter records) added ~10 s to
# each answer even for documents with no voter records in them.
SYSTEM_PROMPT = """\
You answer questions using ONLY the numbered evidence provided.

CRITICAL CITATION RULES:
1. Answer in sentences that state the facts, each followed by its evidence number, e.g. "Residential use is 55.04% [1]." Never reply with a citation alone. NEVER output original paper reference numbers (such as [64], [18]) as citations.
2. If the evidence does not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE.
3. Answer in the language of the question; read Hindi/Devanagari evidence accurately.
4. Do not guess, speculate, apologise or explain your reasoning. Keep the answer under 120 words.
5. If asked what a page or section contains, summarise that evidence."""

# Added only when the evidence holds voter records.
ELECTORAL_RULES = """
6. Each line "- [Serial: ... | EPIC: ... | Voter: ... | Relation: ... | House: ...]" is one person; never mix fields from different lines.
7. For a question about one record (ID, name or serial), give every field of that record.
8. For "which voters", "who lives in" or "list all" questions, list every matching record by name and serial with its citation; if any record matches, never reply INSUFFICIENT_EVIDENCE."""

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
# Vocabularies trained mostly on English split Devanagari and Arabic script into
# a token every character or two, so a Hindi word costs several tokens, not
# ~1.3. Counting those scripts by word underestimated electoral-roll evidence
# several-fold and overran the budget on exactly the slowest prompts.
TOKENS_PER_NON_ASCII_CHAR = 0.5


def estimate_tokens(text: str) -> int:
    """Cheap, backend-agnostic token estimate for budgeting."""
    words = text.split()
    ascii_words = sum(1 for w in words if w.isascii())
    non_ascii_chars = sum(1 for ch in text if not ch.isascii())
    return int(ascii_words * TOKENS_PER_WORD + non_ascii_chars * TOKENS_PER_NON_ASCII_CHAR) + 1


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


def _as_chunks(candidates: Iterable, max_parent_tokens: Optional[int] = None) -> list[Chunk]:
    """Resolve candidates to the text the model should read.

    A child is replaced by its parent (the whole table, section or household)
    for context — unless the parent is larger than `max_parent_tokens`. On a
    CPU model every evidence token costs ~35 ms before the first answer token,
    and a 500-word section in place of the matching paragraph was the largest
    single cost in a local answer.
    """
    chunks = []
    seen_parents = set()
    for item in candidates:
        chunk = item.chunk if isinstance(item, ScoredChunk) else item
        if chunk is not None:
            parent_id = (chunk.metadata or {}).get("parent_id")
            parent_text = (chunk.metadata or {}).get("parent_text")
            if parent_text and max_parent_tokens is not None \
                    and estimate_tokens(parent_text) > max_parent_tokens:
                parent_text = None  # read the child itself

            # Avoid duplicating identical parent tables/sections if multiple
            # child rows matched. Children read on their own are distinct text.
            if parent_id and parent_text and parent_id in seen_parents:
                continue
            if parent_id and parent_text:
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


def _evidence_label(chunk: Chunk) -> str:
    """Where one piece of evidence came from, e.g. "(pmp-2031-report, p. 66) "."""
    parts = [chunk.title] if chunk.title else []
    if chunk.page is not None:
        parts.append(f"p. {chunk.page}")
    return f"({', '.join(parts)}) " if parts else ""


def _spans_documents(chunks: Sequence[Chunk]) -> bool:
    return len({c.title or c.doc_id.split("#", 1)[0] for c in chunks}) > 1


def format_evidence(chunks: Sequence[Chunk], *, labelled: bool = False) -> str:
    """Number the evidence so the model can cite it positionally.

    `labelled` prefixes each item with its document and page. Evidence from a
    folder of several documents otherwise reaches the model as anonymous
    passages, and it cannot tell the Master Plan's page 66 from a roll's.
    """
    return "\n\n".join(
        f"[{i}] {_evidence_label(chunk) if labelled else ''}{chunk.text.strip()}"
        for i, chunk in enumerate(chunks, 1)
    )


def build_prompt(
    question: str,
    candidates: Iterable,
    *,
    evidence_token_budget: int = 3000,
    max_evidence: int = 8,
    system: str = SYSTEM_PROMPT,
    target_lang: Optional[str] = None,
    max_parent_tokens: Optional[int] = None,
) -> BuiltPrompt:
    """Pack the highest-ranked evidence that fits, in rank order.

    Chunks are included whole or not at all: a half-included chunk would let the
    model cite text it never received.
    """
    chunks = _as_chunks(candidates, max_parent_tokens)[:max_evidence]
    labelled = _spans_documents(chunks)

    kept: list[Chunk] = []
    used = 0
    dropped = 0
    for chunk in chunks:
        label = _evidence_label(chunk) if labelled else ""
        cost = estimate_tokens(label + chunk.text) + 4  # marker + separator
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

    # Voter-record rules only when there are voter records for them to govern.
    if system.startswith(SYSTEM_PROMPT) and any(
        "[Serial:" in c.text or "EPIC:" in c.text for c in kept
    ):
        system = f"{system}{ELECTORAL_RULES}"

    prompt = PROMPT_TEMPLATE.format(
        evidence=format_evidence(kept, labelled=labelled) if kept else "(none)",
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
