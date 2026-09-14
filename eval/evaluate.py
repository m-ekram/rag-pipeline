"""Measure retrieval quality, so tuning is evidence-driven rather than vibes.

    python -m eval.evaluate --compare          # baseline vs tuned, side by side
    python -m eval.evaluate --config tuned     # just the shipping config
    python -m eval.evaluate --ablate           # isolate chunking and retrieval components
    python -m eval.evaluate --sweep            # grid over chunk sizes
    python -m eval.evaluate --validate         # is every question even answerable?
    python -m eval.evaluate --compare --embed openai:text-embedding-3-small --save

Metrics, all computed at k = TOP_K:
  hit_rate   fraction of questions where >=1 relevant chunk made the top k.
             This is the "top-5 retrieval relevance" number - if it is low, no
             prompt engineering downstream will save the answer.
  mrr        mean reciprocal rank of the first relevant chunk. Rewards putting
             the right passage at position 1 rather than position 5.
  precision  fraction of the k returned chunks that were relevant. Low
             precision means the model is wading through noise.

A chunk counts as relevant when its source file is listed for the question and,
when `must_contain` is given, the chunk text actually contains one of those
strings. That second condition is what stops a 40-page PDF scoring a hit just
for being the right file.

`--save` writes JSON to eval/results/ with the git commit, the golden set's
hash and the embedding model, so every number quoted in the README can be
traced to the run that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml
from langchain_core.documents import Document

import config
from app.chunking import naive_split, stats, structured_split
from app.loaders import load_directory
from app.providers import embedding_name, get_embeddings
from app.retriever import build_retriever, dense_only_retriever
from app.store import build_index

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"
RESULTS_DIR = Path(__file__).parent / "results"

# The two ends of the tuning story: what you start with, and what you ship.
# "tuned" reads the shipping defaults from config, so it cannot drift from
# what the API actually serves.
CONFIGS = {
    "baseline": {
        "label": "fixed-width chunks, no overlap, no headers, dense-only top-k",
        "splitter": "naive",
        "chunk_size": 1000,
        "chunk_overlap": 0,
        "header_mode": "none",
        "hybrid": False,
        "mmr_lambda": None,
        "rerank": False,
    },
    "tuned": {
        "label": (
            f"structure-aware chunks + overlap + '{config.HEADER_MODE}' headers, "
            f"{'hybrid BM25/dense' if config.USE_HYBRID else 'dense'}"
            f"{', cross-encoder rerank' if config.RERANK else ''}"
        ),
        "splitter": "structured",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "header_mode": config.HEADER_MODE,
        "hybrid": config.USE_HYBRID,
        "mmr_lambda": config.MMR_LAMBDA,
        "rerank": config.RERANK,
    },
}

# Chunking ablations: each needs its own index (different chunk text means
# different vectors), and all share the shipping retriever.
CHUNK_ABLATIONS = {
    "headers:none": {"header_mode": "none"},
    "headers:title": {"header_mode": "title"},
    "headers:path": {"header_mode": "path"},
}

# Retrieval ablations. All share the tuned chunk set, so the embedding cost is
# paid once and each variant differs only in how candidates are ranked. This is
# what separates "hybrid helps" from "we changed five things and it moved".
RETRIEVAL_ABLATIONS = {
    "dense-only": {"hybrid": False, "mmr_lambda": None, "rerank": False},
    "dense+mmr(0.5)": {"hybrid": False, "mmr_lambda": 0.5, "rerank": False},
    "hybrid": {"hybrid": True, "mmr_lambda": 1.0, "rerank": False},
    "hybrid+mmr(0.5)": {"hybrid": True, "mmr_lambda": 0.5, "rerank": False},
    "sparse-heavy": {"hybrid": True, "mmr_lambda": 1.0, "rerank": False, "weights": (0.4, 0.6)},
    "dense+rerank": {"hybrid": False, "mmr_lambda": 1.0, "rerank": True},
    "hybrid+rerank": {"hybrid": True, "mmr_lambda": 1.0, "rerank": True},
}


@contextmanager
def overrides(**values):
    """Temporarily set config attributes; the retriever reads them at build time."""
    old = {name: getattr(config, name) for name in values}
    for name, value in values.items():
        setattr(config, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(config, name, value)


def load_questions() -> list[dict]:
    if not QUESTIONS_FILE.exists():
        raise SystemExit(f"No golden set at {QUESTIONS_FILE}")
    questions = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8")) or []
    for q in questions:
        if "question" not in q or "relevant_sources" not in q:
            raise SystemExit(f"Malformed entry (needs question + relevant_sources): {q}")
    return questions


def is_relevant(doc: Document, question: dict) -> bool:
    source = doc.metadata.get("source", "")
    if not any(source.endswith(s) for s in question["relevant_sources"]):
        return False
    phrases = question.get("must_contain")
    if not phrases:
        return True
    text = doc.page_content.lower()
    return any(p.lower() in text for p in phrases)


def validate(docs: list[Document], questions: list[dict]) -> int:
    """Check every question is answerable before spending time on embeddings.

    A question whose `relevant_sources` do not exist, or whose `must_contain`
    string appears in no chunk of those sources, can never be scored a hit. It
    would silently drag the reported hit_rate down and make the number
    meaningless. Better to fail loudly than to measure the wrong thing.
    """
    chunks = build_chunks(docs, CONFIGS["tuned"])
    by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.metadata.get("source", ""), []).append(chunk.page_content.lower())

    problems = 0
    for q in questions:
        matching = [s for s in by_source if any(s.endswith(r) for r in q["relevant_sources"])]
        if not matching:
            print(f"  UNREACHABLE  no such source {q['relevant_sources']}\n               {q['question']}")
            problems += 1
            continue
        phrases = q.get("must_contain")
        if not phrases:
            continue
        texts = [t for s in matching for t in by_source[s]]
        if not any(p.lower() in t for p in phrases for t in texts):
            print(f"  UNMATCHED    none of {phrases} appear in {matching}\n               {q['question']}")
            problems += 1

    if problems:
        print(f"\n{problems}/{len(questions)} question(s) can never be scored a hit - fix these first.")
    else:
        print(f"All {len(questions)} questions are reachable in the corpus.")
    return problems


def build_chunks(docs: list[Document], cfg: dict) -> list[Document]:
    if cfg["splitter"] == "naive":
        return naive_split(docs, chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    return structured_split(
        docs,
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        add_headers=cfg["header_mode"] != "none",
        header_mode=cfg["header_mode"] if cfg["header_mode"] != "none" else None,
    )


def embed_index(chunks: list[Document], embed: tuple[str, str]):
    provider, model = embed
    return build_index(chunks, show_progress=False, embeddings=get_embeddings(provider, model), model_name=model)


def make_retriever(store, chunks: list[Document], cfg: dict, k: int):
    if not cfg["hybrid"] and cfg["mmr_lambda"] is None and not cfg["rerank"]:
        return dense_only_retriever(store, k)
    with overrides(
        USE_HYBRID=cfg["hybrid"],
        MMR_LAMBDA=1.0 if cfg["mmr_lambda"] is None else cfg["mmr_lambda"],
        RERANK=cfg["rerank"],
        HYBRID_WEIGHTS=cfg.get("weights", config.HYBRID_WEIGHTS),
    ):
        return build_retriever(store, chunks, k)


def score(retriever, questions: list[dict], k: int, verbose: bool = False) -> dict:
    hits = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    by_tag: dict[str, list[int]] = {}
    detail: list[dict] = []

    for question in questions:
        retrieved = retriever.invoke(question["question"])[:k]
        flags = [is_relevant(doc, question) for doc in retrieved]

        hit = any(flags)
        hits += hit
        rank = flags.index(True) + 1 if hit else 0
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        precisions.append(sum(flags) / len(flags) if flags else 0.0)
        for tag in question.get("tags", []):
            by_tag.setdefault(tag, [0, 0])
            by_tag[tag][0] += hit
            by_tag[tag][1] += 1

        detail.append(
            {
                "question": question["question"],
                "tags": question.get("tags", []),
                "hit": hit,
                "first_relevant_rank": rank,
                "retrieved": [
                    {"source": d.metadata.get("source"), "section": d.metadata.get("section"), "relevant": f}
                    for d, f in zip(retrieved, flags)
                ],
            }
        )
        if verbose:
            print(f"    {'HIT ' if hit else 'MISS'} rank={rank or '-':<3} {question['question'][:64]}")

    n = len(questions)
    return {
        "k": k,
        "questions": n,
        "hits": hits,
        "hit_rate": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "precision": sum(precisions) / n,
        "by_tag": {tag: f"{h}/{t}" for tag, (h, t) in sorted(by_tag.items())},
        "detail": detail,
    }


def run_config(name: str, cfg: dict, docs, questions, k: int, embed: tuple[str, str]) -> dict:
    embed = embedding_name(*cfg["embed"].split(":", 1)) if cfg.get("embed") else embed
    print(f"\n=== {name}: {cfg['label']} [{embed[0]}:{embed[1]}] ===")

    chunks = build_chunks(docs, cfg)
    s = stats(chunks)
    print(f"    {s['count']} chunks (median {s['median_chars']} chars) - embedding...")
    store = embed_index(chunks, embed)
    result = score(make_retriever(store, chunks, cfg, k), questions, k, verbose=True)
    result = {"config": name, "label": cfg["label"], "embedding": f"{embed[0]}:{embed[1]}", "chunks": s["count"], **result}
    print(
        f"    -> hit_rate@{k} {result['hit_rate']:.1%} ({result['hits']}/{result['questions']}) | "
        f"MRR {result['mrr']:.3f} | precision@{k} {result['precision']:.1%} | {result['by_tag']}"
    )
    return result


def run_ablations(docs, questions, k: int, embed: tuple[str, str]) -> list[dict]:
    results = []
    tuned = CONFIGS["tuned"]

    print("\nChunking variants (each re-embedded; shipping retriever):")
    for name, spec in CHUNK_ABLATIONS.items():
        cfg = dict(tuned, **spec)
        chunks = build_chunks(docs, cfg)
        store = embed_index(chunks, embed)
        results.append(_ablation_row(name, spec, chunks, score(make_retriever(store, chunks, cfg, k), questions, k), k))

    chunks = build_chunks(docs, tuned)
    print(f"\nRetrieval variants (shared index: {len(chunks)} tuned chunks, embedded once):")
    store = embed_index(chunks, embed)
    for name, spec in RETRIEVAL_ABLATIONS.items():
        cfg = dict(tuned, **spec)
        results.append(_ablation_row(name, spec, chunks, score(make_retriever(store, chunks, cfg, k), questions, k), k))
    for r in results:
        r["embedding"] = f"{embed[0]}:{embed[1]}"
    return results


def _ablation_row(name: str, spec: dict, chunks, result: dict, k: int) -> dict:
    print(
        f"  {name:18} hit@{k} {result['hit_rate']:6.1%}  MRR {result['mrr']:.3f}  "
        f"P@{k} {result['precision']:6.1%}  {result['by_tag']}"
    )
    return {"config": name, "label": json.dumps(spec, default=str), "chunks": len(chunks), **result}


def print_comparison(results: list[dict], k: int) -> None:
    header = f"{'config':<18}{'chunks':>8}{'hit@' + str(k):>10}{'MRR':>10}{'precision':>12}"
    print("\n" + "=" * 74)
    print(header)
    print("-" * 74)
    for r in results:
        print(f"{r['config']:<18}{r['chunks']:>8}{r['hit_rate']:>9.1%}{r['mrr']:>10.3f}{r['precision']:>11.1%}")
    if len(results) >= 2:
        first, last = results[0], results[-1]
        print("-" * 74)
        print(
            f"top-{k} retrieval relevance: {first['hit_rate']:.1%} -> {last['hit_rate']:.1%} "
            f"({(last['hit_rate'] - first['hit_rate']) * 100:+.1f} points, {first['config']} -> {last['config']})"
        )
    print("=" * 74)


def provenance(k: int, embed: tuple[str, str], data_dir: str) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "questions_sha1": hashlib.sha1(QUESTIONS_FILE.read_bytes()).hexdigest(),
        "embedding": f"{embed[0]}:{embed[1]}",
        "k": k,
        "data_dir": data_dir,
        "config": config.summary(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save(kind: str, results: list[dict], meta: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--config", choices=sorted(CONFIGS), help="evaluate a single configuration")
    parser.add_argument("--compare", action="store_true", help="baseline vs tuned (default)")
    parser.add_argument("--sweep", action="store_true", help="grid over chunk sizes and overlaps")
    parser.add_argument("--ablate", action="store_true", help="isolate each chunking and retrieval component")
    parser.add_argument("--embed", help="provider:model for every config, e.g. openai:text-embedding-3-large")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--save", action="store_true", help="write JSON to eval/results/")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="check every question is reachable in the corpus, then exit (no embedding)",
    )
    args = parser.parse_args(argv)

    questions = load_questions()
    docs = load_directory(args.data_dir)
    print(f"{len(questions)} golden questions | {len(docs)} loaded section(s) | k={args.k}")

    if args.validate:
        return 1 if validate(docs, questions) else 0

    embed = embedding_name(*args.embed.split(":", 1)) if args.embed else embedding_name()
    config.require_embed_key(embed[0])
    meta = provenance(args.k, embed, args.data_dir)

    if args.ablate:
        results = run_ablations(docs, questions, args.k, embed)
        if args.save:
            save("ablate", results, meta)
        return 0

    if args.sweep:
        results = []
        for size, overlap in [(500, 75), (800, 120), (1000, 150), (1500, 225)]:
            cfg = dict(CONFIGS["tuned"], chunk_size=size, chunk_overlap=overlap, label=f"tuned, chunk={size}/{overlap}")
            results.append(run_config(f"cs{size}", cfg, docs, questions, args.k, embed))
    elif args.config:
        results = [run_config(args.config, CONFIGS[args.config], docs, questions, args.k, embed)]
    else:
        results = [run_config(name, CONFIGS[name], docs, questions, args.k, embed) for name in ("baseline", "tuned")]

    print_comparison(results, args.k)
    if args.save:
        save("sweep" if args.sweep else "eval", results, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
