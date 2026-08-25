"""Measure retrieval quality, so tuning is evidence-driven rather than vibes.

    python -m eval.evaluate --compare          # baseline vs tuned, side by side
    python -m eval.evaluate --config tuned     # just the shipping config
    python -m eval.evaluate --sweep            # grid over chunk sizes

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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from langchain_core.documents import Document

import config
from app.chunking import naive_split, stats, structured_split
from app.loaders import load_directory
from app.retriever import build_retriever, dense_only_retriever
from app.store import build_index

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"
RESULTS_DIR = Path(__file__).parent / "results"

# The two ends of the tuning story: what you start with, and what you ship.
CONFIGS = {
    "baseline": {
        "label": "fixed-width chunks, no overlap, dense-only top-k",
        "splitter": "naive",
        "chunk_size": 1000,
        "chunk_overlap": 0,
        "headers": False,
        "hybrid": False,
        "mmr": False,
    },
    "tuned": {
        "label": "structure-aware chunks + overlap + headers, hybrid BM25/dense, MMR off",
        "splitter": "structured",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "headers": True,
        "hybrid": True,
        "mmr": True,
    },
}


# Retrieval ablations. All share the tuned chunk set, so the embedding cost is
# paid once and each variant differs only in how candidates are ranked. This is
# what separates "hybrid helps" from "we changed five things and it moved".
ABLATIONS = {
    "dense-only": {"hybrid": False, "mmr": False},
    "dense+mmr": {"hybrid": False, "mmr": True},
    "hybrid": {"hybrid": True, "mmr": False},
    "hybrid+mmr": {"hybrid": True, "mmr": True},
    "hybrid+mmr(l=0.8)": {"hybrid": True, "mmr": True, "lambda": 0.8},
    "hybrid+mmr(l=1.0)": {"hybrid": True, "mmr": True, "lambda": 1.0},
    "sparse-heavy": {"hybrid": True, "mmr": True, "weights": (0.4, 0.6)},
}


def run_ablations(docs: list[Document], questions: list[dict], k: int) -> list[dict]:
    """Vary only the retriever; chunks and vectors are built once and reused."""
    chunks = build_chunks(docs, CONFIGS["tuned"])
    print(f"\nBuilding shared index: {len(chunks)} chunks (embedded once, reused by every variant)")
    store = build_index(chunks, show_progress=False)

    results = []
    original = (config.USE_HYBRID, config.MMR_LAMBDA, config.HYBRID_WEIGHTS)
    try:
        for name, spec in ABLATIONS.items():
            config.USE_HYBRID = spec["hybrid"]
            config.MMR_LAMBDA = spec.get("lambda", 1.0 if not spec["mmr"] else original[1])
            config.HYBRID_WEIGHTS = spec.get("weights", original[2])
            retriever = build_retriever(store, chunks, k) if spec["hybrid"] or spec["mmr"] else dense_only_retriever(store, k)

            hits = 0
            rr: list[float] = []
            prec: list[float] = []
            for q in questions:
                got = retriever.invoke(q["question"])[:k]
                flags = [is_relevant(d, q) for d in got]
                hit = any(flags)
                hits += hit
                rank = flags.index(True) + 1 if hit else 0
                rr.append(1.0 / rank if rank else 0.0)
                prec.append(sum(flags) / len(flags) if flags else 0.0)

            n = len(questions)
            results.append(
                {
                    "config": name,
                    "label": str(spec),
                    "k": k,
                    "questions": n,
                    "chunks": len(chunks),
                    "hit_rate": hits / n,
                    "mrr": sum(rr) / n,
                    "precision": sum(prec) / n,
                    "detail": [],
                }
            )
            print(f"  {name:20} hit@{k} {hits / n:6.1%}  MRR {results[-1]['mrr']:.3f}  P@{k} {results[-1]['precision']:6.1%}")
    finally:
        config.USE_HYBRID, config.MMR_LAMBDA, config.HYBRID_WEIGHTS = original
    return results


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
        add_headers=cfg["headers"],
    )


def make_retriever(store, chunks: list[Document], cfg: dict, k: int):
    if not cfg["hybrid"] and not cfg["mmr"]:
        return dense_only_retriever(store, k)
    original_hybrid, original_mmr = config.USE_HYBRID, config.MMR_LAMBDA
    config.USE_HYBRID = cfg["hybrid"]
    if not cfg["mmr"]:
        config.MMR_LAMBDA = 1.0  # lambda=1 makes MMR equivalent to plain similarity
    try:
        return build_retriever(store, chunks, k)
    finally:
        config.USE_HYBRID, config.MMR_LAMBDA = original_hybrid, original_mmr


def run_config(name: str, cfg: dict, docs: list[Document], questions: list[dict], k: int) -> dict:
    print(f"\n=== {name}: {cfg['label']} ===")

    chunks = build_chunks(docs, cfg)
    s = stats(chunks)
    print(f"    {s['count']} chunks (median {s['median_chars']} chars) - embedding...")

    store = build_index(chunks, show_progress=False)
    retriever = make_retriever(store, chunks, cfg, k)

    hits = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    per_question: list[dict] = []

    for question in questions:
        retrieved = retriever.invoke(question["question"])[:k]
        flags = [is_relevant(doc, question) for doc in retrieved]

        hit = any(flags)
        hits += hit
        rank = flags.index(True) + 1 if hit else 0
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        precisions.append(sum(flags) / len(flags) if flags else 0.0)

        per_question.append(
            {
                "question": question["question"],
                "hit": hit,
                "first_relevant_rank": rank,
                "retrieved": [
                    {
                        "source": d.metadata.get("source"),
                        "page": d.metadata.get("page"),
                        "relevant": f,
                    }
                    for d, f in zip(retrieved, flags)
                ],
            }
        )
        marker = "HIT " if hit else "MISS"
        print(f"    {marker} rank={rank or '-':<3} {question['question'][:64]}")

    n = len(questions)
    result = {
        "config": name,
        "label": cfg["label"],
        "k": k,
        "questions": n,
        "chunks": s["count"],
        "hit_rate": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "precision": sum(precisions) / n,
        "detail": per_question,
    }
    print(
        f"    -> hit_rate@{k} {result['hit_rate']:.1%} | "
        f"MRR {result['mrr']:.3f} | precision@{k} {result['precision']:.1%}"
    )
    return result


def print_comparison(results: list[dict], k: int) -> None:
    header = f"{'config':<12}{'chunks':>8}{'hit@' + str(k):>10}{'MRR':>10}{'precision':>12}"
    print("\n" + "=" * 74)
    print(header)
    print("-" * 74)
    for r in results:
        print(
            f"{r['config']:<12}{r['chunks']:>8}{r['hit_rate']:>9.1%}"
            f"{r['mrr']:>10.3f}{r['precision']:>11.1%}"
        )
    if len(results) >= 2:
        first, last = results[0], results[-1]
        delta = last["hit_rate"] - first["hit_rate"]
        print("-" * 74)
        print(
            f"top-{k} retrieval relevance: {first['hit_rate']:.0%} -> {last['hit_rate']:.0%} "
            f"({delta:+.0%} from {first['config']} to {last['config']})"
        )
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--config", choices=sorted(CONFIGS), help="evaluate a single configuration")
    parser.add_argument("--compare", action="store_true", help="baseline vs tuned (default)")
    parser.add_argument("--sweep", action="store_true", help="grid over chunk sizes and overlaps")
    parser.add_argument("--ablate", action="store_true", help="isolate each retrieval component")
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

    config.require_embed_key()

    if args.ablate:
        results = run_ablations(docs, questions, args.k)
        print_comparison(results, args.k)
        if args.save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out = RESULTS_DIR / f"ablate-{time.strftime('%Y%m%d-%H%M%S')}.json"
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Saved {out}")
        return 0

    if args.sweep:
        results = []
        for size, overlap in [(500, 75), (800, 120), (1000, 150), (1500, 225)]:
            cfg = dict(CONFIGS["tuned"], chunk_size=size, chunk_overlap=overlap)
            cfg["label"] = f"tuned, chunk={size}/{overlap}"
            results.append(run_config(f"cs{size}", cfg, docs, questions, args.k))
    elif args.config:
        results = [run_config(args.config, CONFIGS[args.config], docs, questions, args.k)]
    else:
        results = [
            run_config(name, CONFIGS[name], docs, questions, args.k) for name in ("baseline", "tuned")
        ]

    print_comparison(results, args.k)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
