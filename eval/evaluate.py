"""Measure retrieval quality, so tuning is evidence-driven rather than vibes.

    python -m eval.evaluate --compare          # baseline vs tuned, side by side
    python -m eval.evaluate --config tuned     # just the shipping config
    python -m eval.evaluate --sweep            # grid over chunk sizes
    python -m eval.evaluate --ablate           # isolate each retrieval component
    python -m eval.evaluate --check            # fail if it regressed vs baseline.json

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
BASELINE_FILE = Path(__file__).parent / "baseline.json"
CORPUS_LOCK = Path("data/fastapi/CORPUS.lock.json")

# How far a metric may drift from the checked-in baseline before --check fails.
# Retrieval here is deterministic, so this is slack for a corpus or library
# bump rather than for run-to-run noise.
TOLERANCE = 0.02

METRICS = ("hit_rate", "mrr", "precision")


def corpus_provenance() -> dict:
    """Record which corpus a number was measured on, so it stays falsifiable."""
    if not CORPUS_LOCK.is_file():
        return {}
    lock = json.loads(CORPUS_LOCK.read_text(encoding="utf-8"))
    return {k: lock[k] for k in ("ref", "commit", "file_count") if k in lock}


def summarise(results: list[dict]) -> dict:
    """The part of a run worth checking in: metrics without the per-question dump."""
    return {
        "corpus": corpus_provenance(),
        "embedding_model": config.EMBEDDING_MODEL,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [
            {key: r[key] for key in
             ("config", "label", "k", "questions", "chunks", *METRICS, "by_tag")}
            for r in results
        ],
    }


def check_against_baseline(results: list[dict], tolerance: float) -> int:
    """Compare a fresh run against the checked-in baseline. Returns a failure count."""
    if not BASELINE_FILE.is_file():
        print(f"No baseline at {BASELINE_FILE}. Write one with --save-baseline.")
        return 1

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    expected = {r["config"]: r for r in baseline["results"]}

    if baseline.get("corpus") and corpus_provenance() != baseline["corpus"]:
        print(
            f"  CORPUS   baseline was measured on {baseline['corpus']}, "
            f"this corpus is {corpus_provenance()}"
        )

    failures = 0
    print(f"\n{'config':<12}{'metric':<12}{'baseline':>10}{'now':>10}{'delta':>10}")
    print("-" * 54)
    for result in results:
        name = result["config"]
        if name not in expected:
            print(f"  NEW      {name} is not in the baseline")
            continue
        for metric in METRICS:
            was, now = expected[name][metric], result[metric]
            delta = now - was
            flag = "  FAIL" if delta < -tolerance else ""
            failures += bool(flag)
            print(f"{name:<12}{metric:<12}{was:>10.3f}{now:>10.3f}{delta:>+10.3f}{flag}")

    if failures:
        print(f"\n{failures} metric(s) regressed by more than {tolerance:.3f}.")
    else:
        print(f"\nAll metrics within {tolerance:.3f} of the baseline.")
    return failures


# Retrieval settings are declared per variant, never inherited from ambient
# config. `mmr_lambda: None` means MMR off - lambda 1.0 makes it equivalent to
# plain relevance ranking.
#
# This used to read `spec.get("lambda", ... original[1])`, which took the
# ambient config value. Once MMR_LAMBDA defaulted to 1.0 the "+mmr" variants
# silently ran with MMR *disabled*, making them identical to their non-MMR
# twins and erasing the finding the table exists to document.

# The two ends of the tuning story: what you start with, and what you ship.
CONFIGS = {
    "baseline": {
        "label": "fixed-width chunks, no overlap, dense-only top-k",
        "splitter": "naive",
        "chunk_size": 1000,
        "chunk_overlap": 0,
        "headers": False,
        "hybrid": False,
        "mmr_lambda": None,
    },
    "tuned": {
        "label": "structure-aware chunks + overlap + headers, hybrid BM25/dense, MMR off",
        "splitter": "structured",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "headers": True,
        "hybrid": True,
        "mmr_lambda": None,
    },
}


# Retrieval ablations. All share the tuned chunk set, so the embedding cost is
# paid once and each variant differs only in how candidates are ranked. This is
# what separates "hybrid helps" from "we changed five things and it moved".
ABLATIONS = {
    "dense-only": {"hybrid": False, "mmr_lambda": None},
    "dense+mmr": {"hybrid": False, "mmr_lambda": 0.5},
    "hybrid": {"hybrid": True, "mmr_lambda": None},
    "hybrid+mmr": {"hybrid": True, "mmr_lambda": 0.5},
    "hybrid+mmr(l=0.8)": {"hybrid": True, "mmr_lambda": 0.8},
    "sparse-heavy": {"hybrid": True, "mmr_lambda": None, "weights": (0.4, 0.6)},
}


def retriever_for(store, chunks: list[Document], spec: dict, k: int):
    """Build a retriever from an explicit spec, touching no global state."""
    mmr_lambda = spec.get("mmr_lambda")
    if not spec["hybrid"] and mmr_lambda is None:
        return dense_only_retriever(store, k)
    return build_retriever(
        store,
        chunks,
        k,
        use_hybrid=spec["hybrid"],
        mmr_lambda=1.0 if mmr_lambda is None else mmr_lambda,
        weights=spec.get("weights"),
    )


def score(retriever, questions: list[dict], k: int) -> tuple[float, float, float, list[dict]]:
    """hit_rate, MRR and precision at k, plus the per-question detail."""
    hits = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    detail: list[dict] = []

    for question in questions:
        retrieved = retriever.invoke(question["question"])[:k]
        flags = [is_relevant(doc, question) for doc in retrieved]

        hit = any(flags)
        hits += hit
        rank = flags.index(True) + 1 if hit else 0
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        precisions.append(sum(flags) / len(flags) if flags else 0.0)

        detail.append(
            {
                "question": question["question"],
                "tags": question.get("tags", []),
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

    n = len(questions)
    return hits / n, sum(reciprocal_ranks) / n, sum(precisions) / n, detail


def by_tag(detail: list[dict]) -> dict[str, str]:
    """Hits per failure class. An average over only one class is misleading."""
    buckets: dict[str, list[bool]] = {}
    for entry in detail:
        for tag in entry.get("tags") or ["untagged"]:
            buckets.setdefault(tag, []).append(entry["hit"])
    return {tag: f"{sum(hits)}/{len(hits)}" for tag, hits in sorted(buckets.items())}


def run_ablations(docs: list[Document], questions: list[dict], k: int) -> list[dict]:
    """Vary only the retriever; chunks and vectors are built once and reused."""
    chunks = build_chunks(docs, CONFIGS["tuned"])
    print(f"\nBuilding shared index: {len(chunks)} chunks (embedded once, reused by every variant)")
    store = build_index(chunks, show_progress=False)

    results = []
    for name, spec in ABLATIONS.items():
        hit_rate, mrr, precision, detail = score(
            retriever_for(store, chunks, spec, k), questions, k
        )
        results.append(
            {
                "config": name,
                "label": str(spec),
                "k": k,
                "questions": len(questions),
                "chunks": len(chunks),
                "hit_rate": hit_rate,
                "mrr": mrr,
                "precision": precision,
                "by_tag": by_tag(detail),
                "detail": detail,
            }
        )
        print(f"  {name:20} hit@{k} {hit_rate:6.1%}  MRR {mrr:.3f}  P@{k} {precision:6.1%}")
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


def run_config(name: str, cfg: dict, docs: list[Document], questions: list[dict], k: int) -> dict:
    print(f"\n=== {name}: {cfg['label']} ===")

    chunks = build_chunks(docs, cfg)
    s = stats(chunks)
    print(f"    {s['count']} chunks (median {s['median_chars']} chars) - embedding...")

    store = build_index(chunks, show_progress=False)
    hit_rate, mrr, precision, detail = score(
        retriever_for(store, chunks, cfg, k), questions, k
    )

    for entry in detail:
        marker = "HIT " if entry["hit"] else "MISS"
        rank = entry["first_relevant_rank"] or "-"
        print(f"    {marker} rank={rank:<3} {entry['question'][:64]}")

    result = {
        "config": name,
        "label": cfg["label"],
        "k": k,
        "questions": len(questions),
        "chunks": s["count"],
        "hit_rate": hit_rate,
        "mrr": mrr,
        "precision": precision,
        "by_tag": by_tag(detail),
        "detail": detail,
    }
    print(
        f"    -> hit_rate@{k} {result['hit_rate']:.1%} | "
        f"MRR {result['mrr']:.3f} | precision@{k} {result['precision']:.1%}"
        f"  ({', '.join(f'{t} {v}' for t, v in result['by_tag'].items())})"
    )
    return result


def print_comparison(results: list[dict], k: int) -> None:
    header = f"{'config':<20}{'chunks':>8}{'hit@' + str(k):>10}{'MRR':>10}{'precision':>12}"
    print("\n" + "=" * 74)
    print(header)
    print("-" * 82)
    for r in results:
        print(
            f"{r['config']:<20}{r['chunks']:>8}{r['hit_rate']:>9.1%}"
            f"{r['mrr']:>10.3f}{r['precision']:>11.1%}"
        )
    # Only meaningful when the first row is the untuned starting point; for an
    # ablation grid the first and last rows are unrelated variants.
    if len(results) >= 2 and results[0]["config"] == "baseline":
        first, last = results[0], results[-1]
        delta = last["hit_rate"] - first["hit_rate"]
        print("-" * 82)
        print(
            f"top-{k} retrieval relevance: {first['hit_rate']:.0%} -> {last['hit_rate']:.0%} "
            f"({delta:+.0%} from {first['config']} to {last['config']})"
        )
    print("=" * 82)


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any metric regressed against the checked-in eval/baseline.json",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="overwrite eval/baseline.json with this run",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help=f"allowed regression per metric for --check (default {TOLERANCE})",
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
        prefix = "ablate"
    elif args.sweep:
        results = []
        for size, overlap in [(500, 75), (800, 120), (1000, 150), (1500, 225)]:
            cfg = dict(CONFIGS["tuned"], chunk_size=size, chunk_overlap=overlap)
            cfg["label"] = f"tuned, chunk={size}/{overlap}"
            results.append(run_config(f"cs{size}", cfg, docs, questions, args.k))
        prefix = "sweep"
    elif args.config:
        results = [run_config(args.config, CONFIGS[args.config], docs, questions, args.k)]
        prefix = "eval"
    else:
        results = [
            run_config(name, CONFIGS[name], docs, questions, args.k) for name in ("baseline", "tuned")
        ]
        prefix = "eval"

    print_comparison(results, args.k)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved {out}")

    if args.save_baseline:
        BASELINE_FILE.write_text(
            json.dumps(summarise(results), indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote baseline to {BASELINE_FILE}")

    if args.check:
        return 1 if check_against_baseline(results, args.tolerance) else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
