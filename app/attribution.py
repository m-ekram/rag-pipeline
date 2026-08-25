"""Post-hoc citation attribution.

Large hosted models will happily follow "cite [1] after every claim". Small
local models (3B and below) mostly will not - they answer correctly and
groundedly, then omit the markers entirely. Verified here: Qwen2.5-3B obeys a
plain system instruction ("answer in one word" -> "Green") but produced zero
citations across four prompt formulations.

Rather than degrade the feature for local users, attribution is computed:
each sentence of the answer is embedded and matched against the passages that
were actually retrieved, and the best match above a similarity threshold
becomes its citation.

What a citation means here is therefore "this sentence is semantically closest
to passage N, above a confidence floor" - not "the model asserted N". That is a
weaker claim than a model-emitted citation, so the threshold is deliberately
conservative: a sentence with no good match gets no marker rather than a
misleading one.
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

# Split on sentence enders followed by whitespace + a capital/backtick/digit.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z`\d*\-])")
_FENCE = re.compile(r"```")
_HAS_CITATION = re.compile(r"\[\d+\]")
# Lines we must not append a marker to: headings, code, table rows, bare list
# bullets with no prose.
_SKIP_LINE = re.compile(r"^\s*(?:#{1,6}\s|\||```|\s*[-*]\s*$|\d+\.\s*$)")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _split_outside_code(text: str) -> list[tuple[str, bool]]:
    """Yield (segment, is_code) so fenced blocks are never annotated."""
    parts: list[tuple[str, bool]] = []
    in_code = False
    for block in _FENCE.split(text):
        parts.append((block, in_code))
        in_code = not in_code
    return parts


def attribute(answer: str, docs: list[Document], embeddings) -> str:
    """Append [n] markers to sentences that clearly match a retrieved passage."""
    if not answer.strip() or not docs:
        return answer
    if _HAS_CITATION.search(answer):
        return answer  # the model already cited; leave its judgement alone

    try:
        doc_vectors = embeddings.embed_documents([d.page_content for d in docs])
    except Exception as exc:
        logger.warning("Attribution skipped (embedding failed: %s)", exc)
        return answer

    out_parts: list[str] = []
    for segment, is_code in _split_outside_code(answer):
        if is_code:
            out_parts.append(segment)
            continue
        out_parts.append(_annotate_prose(segment, doc_vectors, embeddings))

    return "```".join(out_parts)


def _annotate_prose(text: str, doc_vectors: list[list[float]], embeddings) -> str:
    lines_out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) < config.CITE_MIN_CHARS or _SKIP_LINE.match(line):
            lines_out.append(line)
            continue

        sentences = _SENTENCE_SPLIT.split(line)
        candidates = [s for s in sentences if len(s.strip()) >= config.CITE_MIN_CHARS]
        if not candidates:
            lines_out.append(line)
            continue

        try:
            vectors = embeddings.embed_documents(candidates)
        except Exception:
            lines_out.append(line)
            continue

        scored: dict[str, str] = {}
        for sentence, vector in zip(candidates, vectors):
            sims = [_cosine(vector, dv) for dv in doc_vectors]
            best = max(range(len(sims)), key=lambda i: sims[i])
            if sims[best] >= config.CITE_THRESHOLD:
                scored[sentence] = f"[{best + 1}]"

        rebuilt = []
        for sentence in sentences:
            marker = scored.get(sentence)
            if not marker:
                rebuilt.append(sentence)
                continue
            # Put the marker before the trailing punctuation, like a human would.
            match = re.search(r"([.!?:])(\s*)$", sentence)
            if match:
                rebuilt.append(f"{sentence[: match.start()]} {marker}{match.group(1)}")
            else:
                rebuilt.append(f"{sentence} {marker}")
        lines_out.append(" ".join(rebuilt))

    return "\n".join(lines_out)
