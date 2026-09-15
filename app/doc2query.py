"""Document expansion (doc2query): index each chunk with questions it answers.

A paraphrased question shares little vocabulary with the passage that answers
it - "how do I serve CSS and images" against a page that only ever says
"static files". Predicting such questions at index time, from the chunk
itself, moves that gap to where there is a whole passage to read instead of
one short query (Nogueira et al., "Document Expansion by Query Prediction").

The expansion is indexing-only. Dense vectors and the lexical leg see
chunk + generated questions; what is returned to the LLM, cited, and scored
by the eval is the original chunk text, unchanged. Generated text never
enters page_content, so it can never make a chunk *look* relevant.

    python -m app.doc2query            # precompute for the corpus (resumable)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

PROMPT = (
    "Below is a passage from a software library's documentation.\n\n{passage}\n\n"
    "Write {n} different questions a developer could ask that this passage answers. "
    "Phrase them the way a user would, in everyday words, instead of repeating the "
    "passage's exact terms. One question per line, no numbering."
)
_MAX_PASSAGE_CHARS = 1800
_LINE_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def chunk_body(text: str) -> str:
    """The chunk without its contextual "[...]" header line."""
    if text.startswith("[") and "]\n" in text:
        return text.split("]\n", 1)[1]
    return text


def parse_questions(output: str, n: int) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()
    for line in output.splitlines():
        line = _LINE_PREFIX.sub("", line).strip().strip('"').strip()
        if len(line) < 10 or line.lower().startswith(("here are", "questions:")):
            continue
        if line.casefold() not in seen:  # a case-only difference is not a new phrasing
            seen.add(line.casefold())
            questions.append(line)
    return questions[:n]


class ExpansionCache:
    """Generated questions keyed by a hash of the chunk body - shared by every
    header mode, since the body is the same."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, list[str]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                        self.entries[record["k"]] = record["q"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # a torn final line from an interrupted run

    def add(self, key: str, questions: list[str]) -> None:
        self.entries[key] = questions
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": key, "q": questions}) + "\n")


def _cache_path(n: int) -> Path:
    slug = re.sub(r"[^\w.-]", "_", Path(config.GEN_MODEL_DIR).name)
    return Path(config.EMBED_CACHE_DIR) / f"doc2query-{slug}-n{n}.jsonl"


def expand_chunks(chunks: list[Document], generator=None, n: int | None = None, show_progress: bool = True) -> list[Document]:
    """Attach `metadata["expansion"]` (newline-joined questions) to every chunk.

    Cached per chunk body, so an interrupted run resumes and re-chunking only
    generates for bodies it has not seen.
    """
    n = n or config.DOC2QUERY_N
    cache = ExpansionCache(_cache_path(n))
    keys = [hashlib.sha1(chunk_body(c.page_content).encode("utf-8")).hexdigest() for c in chunks]
    todo = [i for i, key in enumerate(keys) if key not in cache.entries]
    if todo:
        if generator is None:
            from app.generator import get_generator

            generator = get_generator()
        started = time.time()
        for done, i in enumerate(todo, start=1):
            key = keys[i]
            if key in cache.entries:  # duplicate body later in the list
                continue
            passage = chunks[i].page_content[:_MAX_PASSAGE_CHARS]
            output = generator.chat(PROMPT.format(passage=passage, n=n), max_new_tokens=32 * n)
            cache.add(key, parse_questions(output, n))
            if show_progress and (done % 25 == 0 or done == len(todo)):
                rate = done / (time.time() - started)
                print(f"  doc2query {done}/{len(todo)}  {rate * 60:.1f} chunks/min  ETA {(len(todo) - done) / rate / 60:.0f} min", flush=True)
    for chunk, key in zip(chunks, keys):
        chunk.metadata["expansion"] = "\n".join(cache.entries.get(key, []))
    return chunks


def indexed_text(doc: Document, base: str | None = None) -> str:
    """What an index sees for a chunk: its text (or `base`) plus any expansion."""
    text = doc.page_content if base is None else base
    expansion = doc.metadata.get("expansion")
    return f"{text}\n{expansion}" if expansion else text


def main(argv: list[str] | None = None) -> int:
    from app.chunking import structured_split
    from app.loaders import load_directory

    parser = argparse.ArgumentParser(description="Precompute doc2query expansions for the corpus.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("-n", type=int, default=config.DOC2QUERY_N, help="questions per chunk")
    parser.add_argument("--limit", type=int, default=0, help="only the first N chunks (smoke test)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    chunks = structured_split(load_directory(args.data_dir))
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"{len(chunks)} chunks -> {_cache_path(args.n)}")
    expand_chunks(chunks, n=args.n)
    sample = next((c for c in chunks if c.metadata.get("expansion")), None)
    if sample:
        print(f"\nexample - {sample.metadata.get('source')} / {sample.metadata.get('section')}:\n{sample.metadata['expansion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
