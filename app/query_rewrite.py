"""Query rewriting: search with several phrasings of a question, fuse the results.

The user's wording is one sample from many ways to ask the same thing, and a
paraphrase miss is exactly the case where that sample shares little with the
source text. Each question is expanded into:

  - REWRITE_N paraphrases, in different words
  - one hypothetical answer passage written in documentation style (HyDE,
    Gao et al. 2022), which embeds closer to real passages than a question does

Every variant, plus the original, runs through the normal retriever and the
rankings are fused by reciprocal rank fusion (c=60, equal weights - no knob).

Cost: REWRITE_N + 1 local generations per question, seconds each on CPU.
Rewrites are cached by question text, so repeated eval runs are cheap - but a
live query pays the full cost, which is why this is off by default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

import config
from app.doc2query import parse_questions

PARAPHRASE_PROMPT = (
    "Rewrite this question about a software library's documentation in {n} different ways. "
    "Use different words and name the technical concept it is really about, if you can. "
    "One rewrite per line, no numbering.\n\nQuestion: {question}"
)
HYDE_PROMPT = (
    "Write a short passage (2-3 sentences) in the style of technical documentation that "
    "answers this question. Name the specific functions, classes or settings involved.\n\n"
    "Question: {question}"
)
RRF_K = 60


class RewriteCache:
    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, list[str]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                        self.entries[record["k"]] = record["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def add(self, key: str, variants: list[str]) -> None:
        self.entries[key] = variants
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": key, "v": variants}) + "\n")


def _cache() -> RewriteCache:
    slug = Path(config.GEN_MODEL_DIR).name
    return RewriteCache(Path(config.EMBED_CACHE_DIR) / f"rewrite-{slug}-n{config.REWRITE_N}.jsonl")


def query_variants(question: str, generator=None, cache: RewriteCache | None = None) -> list[str]:
    """[original, paraphrase 1..n, hypothetical passage]."""
    cache = cache or _cache()
    key = hashlib.sha1(question.encode("utf-8")).hexdigest()
    if key not in cache.entries:
        if generator is None:
            from app.generator import get_generator

            generator = get_generator()
        n = config.REWRITE_N
        paraphrases = parse_questions(generator.chat(PARAPHRASE_PROMPT.format(n=n, question=question), 40 * n), n)
        passage = generator.chat(HYDE_PROMPT.format(question=question), 120)
        cache.add(key, paraphrases + ([passage] if passage else []))
    return [question] + cache.entries[key]


def _key(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.page_content


class RewriteFusionRetriever(BaseRetriever):
    """Runs `base` once per query variant and fuses the rankings (RRF)."""

    base: Any
    k: int = 5
    generator: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for variant in query_variants(query, self.generator):
            for rank, doc in enumerate(self.base.invoke(variant), start=1):
                key = _key(doc)
                docs.setdefault(key, doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [docs[key] for key in ordered[: self.k]]
