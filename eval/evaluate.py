"""Measure retrieval quality, so tuning is evidence-driven rather than vibes.

    python -m eval.evaluate --compare          # baseline vs tuned, side by side
    python -m eval.evaluate --config tuned     # just the shipping config
    python -m eval.evaluate --ablate           # isolate chunking and retrieval components
    python -m eval.evaluate --sweep            # grid over chunk sizes
    python -m eval.evaluate --validate         # is every question even answerable?
    python -m eval.evaluate --compare --questions eval/questions_heldout.yaml --save
    python -m eval.evaluate --diff eval/results/A.json#tuned eval/results/B.json#hybrid+rerank
    python -m eval.evaluate --compare --embed openai:text-embedding-3-small --save

Two question sets:
  eval/questions.yaml          dev set - tune against this freely
  eval/questions_heldout.yaml  held-out set - frozen before tuning, scored once
                               at the end. The headline number comes from here.

Metrics, all computed at k = TOP_K:
  hit_rate   fraction of questions where >=1 relevant chunk made the top k.
             This is the "top-5 retrieval relevance" number - if it is low, no
             prompt engineering downstream will save the answer. Reported with
             a Wilson 95% interval: on 36-50 questions one question moves the
             number 2-3 points, and the interval says so.
  hit@1/@3   same, at ranks 1 and 3 - more sensitive tie-breakers than hit@5.
  mrr        mean reciprocal rank of the first relevant chunk. Rewards putting
             the right passage at position 1 rather than position 5.
  precision  fraction of the k returned chunks that were relevant. Low
             precision means the model is wading through noise.

A chunk counts as relevant when its source file is listed for the question and,
when `must_contain` is given, the chunk text actually contains one of those
strings. That second condition is what stops a 40-page PDF scoring a hit just
for being the right file.

`--save` writes JSON to eval/results/ with the git commit, the question set's
hash and the embedding model, so every number quoted in the README can be
traced to the run that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
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
from app.retriever import build_retriever, build_sparse, dense_only_retriever
from app.store import build_index

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"
RESULTS_DIR = Path(__file__).parent / "results"

# The two ends of the tuning story: what you start with, and what you ship.
# "tuned" reads the shipping defaults from config, so it cannot drift from
# what the API actually serves.
CONFIGS = {
    "baseline": {
        "label": "fixed-width chunks, no overlap, no headers, dense-only top-k",
        # Pinned to the original pipeline's embedding model, so changing the
        # tuned embedding can never silently move the "from" figure.
        "embed": "local:BAAI/bge-small-en-v1.5",
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
    "headers:path-clean": {"header_mode": "path-clean"},
}

_BGE_RERANK = "BAAI/bge-reranker-base"
_COLBERT = "answerdotai/answerai-colbert-small-v1"
_MINILM = "Xenova/ms-marco-MiniLM-L-6-v2"

# Retrieval ablations. All share the tuned chunk set, so the embedding cost is
# paid once and each variant differs only in how candidates are ranked. This is
# what separates "hybrid helps" from "we changed five things and it moved".
RETRIEVAL_ABLATIONS = {
    "dense-only": {"hybrid": False, "mmr_lambda": None, "rerank": False},
    "dense+mmr(0.5)": {"hybrid": False, "mmr_lambda": 0.5, "rerank": False},
    "hybrid": {"hybrid": True, "mmr_lambda": 1.0, "rerank": False},
    "hybrid+mmr(0.5)": {"hybrid": True, "mmr_lambda": 0.5, "rerank": False},
    "sparse-heavy": {"hybrid": True, "mmr_lambda": 1.0, "rerank": False, "weights": (0.4, 0.6)},
    "dense+rerank": {"hybrid": False, "mmr_lambda": 1.0, "rerank": True, "rerank_model": _BGE_RERANK, "rerank_fusion": False},
    "hybrid+rerank": {"hybrid": True, "mmr_lambda": 1.0, "rerank": True, "rerank_model": _BGE_RERANK, "rerank_fusion": False},
    "hybrid+rerank+fusion": {"hybrid": True, "mmr_lambda": 1.0, "rerank": True, "rerank_model": _BGE_RERANK, "rerank_fusion": True},
    "hybrid+colbert": {"hybrid": True, "mmr_lambda": 1.0, "rerank": True, "rerank_model": _COLBERT, "rerank_fusion": False},
    "hybrid+colbert+fusion": {"hybrid": True, "mmr_lambda": 1.0, "rerank": True, "rerank_model": _COLBERT, "rerank_fusion": True},
    "hybrid+minilm+fusion": {"hybrid": True, "mmr_lambda": 1.0, "rerank": True, "rerank_model": _MINILM, "rerank_fusion": True},
    "hybrid-splade": {"hybrid": True, "mmr_lambda": 1.0, "rerank": False, "sparse": "splade"},
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


def split_name(path: Path) -> str:
    """questions.yaml -> dev, questions_heldout.yaml -> heldout."""
    stem = path.stem
    return "dev" if stem == "questions" else stem.removeprefix("questions_")


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No question set at {path}")
    questions = yaml.safe_load(path.read_text(encoding="utf-8")) or []
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


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion that stays sane at small n and near 100%."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def lint(docs: list[Document], questions: list[dict]) -> list[str]:
    """Catch questions that would test the wrong thing.

    paraphrase  must not quote its must_contain strings or its page's title -
                otherwise it is an exact-match question wearing a disguise.
    exact       must name at least one must_contain identifier verbatim.
    """
    titles: dict[str, str] = {}
    for doc in docs:
        match = _H1.search(doc.page_content)
        if match:
            titles.setdefault(doc.metadata.get("source", ""), match.group(1).strip().lower())

    issues = []
    for q in questions:
        text = q["question"].lower()
        phrases = [p.lower() for p in q.get("must_contain") or []]
        tags = set(q.get("tags", []))
        if "paraphrase" in tags:
            quoted = [p for p in phrases if p in text]
            page_titles = [t for s, t in titles.items() if any(s.endswith(r) for r in q["relevant_sources"])]
            named = [t for t in page_titles if t in text]
            if quoted or named:
                issues.append(f"  LINT  paraphrase quotes {quoted or named}: {q['question']}")
        if "exact" in tags and phrases and not any(p in text for p in phrases):
            issues.append(f"  LINT  exact names none of {q['must_contain']}: {q['question']}")
    return issues


def validate(docs: list[Document], questions: list[dict], strict: bool = False) -> int:
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

    issues = lint(docs, questions)
    for issue in issues:
        print(issue)
    if strict:
        problems += len(issues)

    if problems:
        print(f"\n{problems} problem(s) across {len(questions)} question(s) - fix these first.")
    else:
        print(f"All {len(questions)} questions are reachable in the corpus" + (" and lint-clean." if strict else "."))
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
        RERANK_MODEL=cfg.get("rerank_model", config.RERANK_MODEL),
        RERANK_FUSION=cfg.get("rerank_fusion", config.RERANK_FUSION),
        SPARSE=cfg.get("sparse", config.SPARSE),
        HYBRID_WEIGHTS=cfg.get("weights", config.HYBRID_WEIGHTS),
    ):
        return build_retriever(store, chunks, k)


def depth_report(store, chunks: list[Document], cfg: dict, result: dict, questions: list[dict], depth: int) -> None:
    """For every miss: where does the first relevant chunk sit in each leg's top `depth`?

    A miss at rank 9 is a ranking problem a re-ranker can fix; a miss at rank 0
    (absent from the top `depth` everywhere) is a recall problem no re-ranker
    can reach.
    """
    misses = [q for q, d in zip(questions, result["detail"]) if not d["hit"]]
    if not misses:
        return
    legs = {"dense": dense_only_retriever(store, depth)}
    if cfg["hybrid"]:
        legs["bm25"] = build_sparse(chunks).model_copy(update={"k": depth})
    legs["pipeline"] = make_retriever(store, chunks, dict(cfg, rerank=False), depth)
    print(f"\n    first relevant rank within top {depth} (0 = absent):")
    print("    " + "".join(f"{name:>10}" for name in legs) + "   question")
    for q in misses:
        ranks = []
        for leg in legs.values():
            got = leg.invoke(q["question"])[:depth]
            ranks.append(next((i for i, d in enumerate(got, 1) if is_relevant(d, q)), 0))
        print("    " + "".join(f"{r:>10}" for r in ranks) + f"   {q['question'][:60]}")


def score(retriever, questions: list[dict], k: int, verbose: bool = False) -> dict:
    ranks: list[int] = []
    precisions: list[float] = []
    latencies: list[float] = []
    by_tag: dict[str, list[int]] = {}
    detail: list[dict] = []

    for question in questions:
        started = time.perf_counter()
        retrieved = retriever.invoke(question["question"])[:k]
        latencies.append((time.perf_counter() - started) * 1000)
        flags = [is_relevant(doc, question) for doc in retrieved]

        rank = flags.index(True) + 1 if any(flags) else 0
        ranks.append(rank)
        precisions.append(sum(flags) / len(flags) if flags else 0.0)
        for tag in question.get("tags", []):
            by_tag.setdefault(tag, [0, 0])
            by_tag[tag][0] += bool(rank)
            by_tag[tag][1] += 1

        detail.append(
            {
                "question": question["question"],
                "tags": question.get("tags", []),
                "hit": bool(rank),
                "first_relevant_rank": rank,
                "retrieved": [
                    {"source": d.metadata.get("source"), "section": d.metadata.get("section"), "relevant": f}
                    for d, f in zip(retrieved, flags)
                ],
            }
        )
        if verbose:
            print(f"    {'HIT ' if rank else 'MISS'} rank={rank or '-':<3} {question['question'][:64]}")

    n = len(questions)
    hits = sum(1 for r in ranks if r)
    lo, hi = wilson(hits, n)
    latencies.sort()
    return {
        "k": k,
        "questions": n,
        "hits": hits,
        "hit_rate": hits / n,
        "hit_rate_ci95": [round(lo, 4), round(hi, 4)],
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if 0 < r <= 3) / n,
        "mrr": sum(1.0 / r for r in ranks if r) / n,
        "precision": sum(precisions) / n,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1),
            "p95": round(latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)], 1),
        },
        "by_tag": {tag: f"{h}/{t}" for tag, (h, t) in sorted(by_tag.items())},
        "detail": detail,
    }


def _summary_line(r: dict) -> str:
    lo, hi = r["hit_rate_ci95"]
    return (
        f"hit@{r['k']} {r['hit_rate']:6.1%} ({r['hits']}/{r['questions']}, 95% CI {lo:.0%}-{hi:.0%}) | "
        f"hit@1 {r['hit@1']:.0%} hit@3 {r['hit@3']:.0%} | MRR {r['mrr']:.3f} | P@{r['k']} {r['precision']:.1%} | "
        f"{r['latency_ms']['mean']:.0f} ms | {r['by_tag']}"
    )


def run_config(name: str, cfg: dict, docs, questions, k: int, embed: tuple[str, str], depth: int = 0) -> dict:
    embed = embedding_name(*cfg["embed"].split(":", 1)) if cfg.get("embed") else embed
    print(f"\n=== {name}: {cfg['label']} [{embed[0]}:{embed[1]}] ===")

    chunks = build_chunks(docs, cfg)
    s = stats(chunks)
    print(f"    {s['count']} chunks (median {s['median_chars']} chars) - embedding...")
    store = embed_index(chunks, embed)
    result = score(make_retriever(store, chunks, cfg, k), questions, k, verbose=True)
    result = {"config": name, "label": cfg["label"], "embedding": f"{embed[0]}:{embed[1]}", "chunks": s["count"], **result}
    print(f"    -> {_summary_line(result)}")
    if depth:
        depth_report(store, chunks, cfg, result, questions, depth)
    return result


def run_ablations(docs, questions, k: int, embed: tuple[str, str], only: set[str] | None = None) -> list[dict]:
    """`only` restricts the run to the named variants (all when None)."""
    results = []
    tuned = CONFIGS["tuned"]
    wanted = lambda name: only is None or name in only  # noqa: E731

    if any(wanted(n) for n in CHUNK_ABLATIONS):
        print("\nChunking variants (each re-embedded; shipping retriever):")
    for name, spec in CHUNK_ABLATIONS.items():
        if not wanted(name):
            continue
        cfg = dict(tuned, **spec)
        chunks = build_chunks(docs, cfg)
        store = embed_index(chunks, embed)
        results.append(_ablation_row(name, spec, chunks, score(make_retriever(store, chunks, cfg, k), questions, k)))

    retrieval = {n: s for n, s in RETRIEVAL_ABLATIONS.items() if wanted(n)}
    if not retrieval:
        return results
    chunks = build_chunks(docs, tuned)
    print(f"\nRetrieval variants (shared index: {len(chunks)} tuned chunks, embedded once):")
    store = embed_index(chunks, embed)
    for name, spec in retrieval.items():
        cfg = dict(tuned, **spec)
        results.append(_ablation_row(name, spec, chunks, score(make_retriever(store, chunks, cfg, k), questions, k)))
    for r in results:
        r["embedding"] = f"{embed[0]}:{embed[1]}"
    return results


def _ablation_row(name: str, spec: dict, chunks, result: dict) -> dict:
    print(f"  {name:18} {_summary_line(result)}")
    return {"config": name, "label": json.dumps(spec, default=str), "chunks": len(chunks), **result}


def print_comparison(results: list[dict], k: int) -> None:
    header = f"{'config':<18}{'chunks':>7}{'hit@' + str(k):>9}{'95% CI':>13}{'hit@1':>8}{'hit@3':>8}{'MRR':>8}{'P@' + str(k):>8}"
    print("\n" + "=" * 79)
    print(header)
    print("-" * 79)
    for r in results:
        lo, hi = r["hit_rate_ci95"]
        print(
            f"{r['config']:<18}{r['chunks']:>7}{r['hit_rate']:>8.1%}{f'{lo:.0%}-{hi:.0%}':>13}"
            f"{r['hit@1']:>8.0%}{r['hit@3']:>8.0%}{r['mrr']:>8.3f}{r['precision']:>8.1%}"
        )
    if len(results) >= 2:
        first, last = results[0], results[-1]
        print("-" * 79)
        print(
            f"top-{k} retrieval relevance: {first['hit_rate']:.1%} -> {last['hit_rate']:.1%} "
            f"({(last['hit_rate'] - first['hit_rate']) * 100:+.1f} points, {first['config']} -> {last['config']})"
        )
    print("=" * 79)


def _pick(spec: str) -> tuple[str, dict]:
    """'path/to/run.json#config' -> that config's result (default: last one)."""
    path, _, name = spec.partition("#")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data["results"] if isinstance(data, dict) else data
    if not name:
        return results[-1]["config"], results[-1]
    for r in results:
        if r["config"] == name:
            return name, r
    raise SystemExit(f"No config {name!r} in {path}; have {[r['config'] for r in results]}")


def diff(spec_a: str, spec_b: str) -> int:
    """Per-question flips between two saved runs.

    A net "+1" can be "+2 -1"; at this sample size the flips are the evidence,
    not the averages.
    """
    name_a, a = _pick(spec_a)
    name_b, b = _pick(spec_b)
    hits_a = {d["question"]: d for d in a["detail"]}
    hits_b = {d["question"]: d for d in b["detail"]}
    common = [q for q in hits_a if q in hits_b]
    if len(common) != len(hits_a) or len(common) != len(hits_b):
        print(f"warning: runs cover different questions ({len(hits_a)} vs {len(hits_b)}; {len(common)} shared)")

    gained = [q for q in common if hits_b[q]["hit"] and not hits_a[q]["hit"]]
    lost = [q for q in common if hits_a[q]["hit"] and not hits_b[q]["hit"]]
    moved = [
        (q, hits_a[q]["first_relevant_rank"], hits_b[q]["first_relevant_rank"])
        for q in common
        if hits_a[q]["hit"] and hits_b[q]["hit"] and hits_a[q]["first_relevant_rank"] != hits_b[q]["first_relevant_rank"]
    ]

    print(f"{name_a}  ->  {name_b}")
    print(f"  hit@{a['k']}: {a['hits']}/{a['questions']} -> {b['hits']}/{b['questions']}   MRR {a['mrr']:.3f} -> {b['mrr']:.3f}")
    print(f"\n  gained ({len(gained)}):")
    for q in gained:
        print(f"    + {q}")
    print(f"  lost ({len(lost)}):")
    for q in lost:
        print(f"    - {q}")
    better = sum(1 for _, ra, rb in moved if rb < ra)
    print(f"  rank changes among shared hits: {better} better, {len(moved) - better} worse")
    return 0


def provenance(k: int, embed: tuple[str, str], data_dir: str, questions_path: Path) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "questions_file": questions_path.name,
        "questions_sha1": hashlib.sha1(questions_path.read_bytes()).hexdigest(),
        "embedding": f"{embed[0]}:{embed[1]}",
        "k": k,
        "data_dir": data_dir,
        "config": config.summary(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save(kind: str, split: str, results: list[dict], meta: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{kind}-{split}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--config", choices=sorted(CONFIGS), help="evaluate a single configuration")
    parser.add_argument("--compare", action="store_true", help="baseline vs tuned (default)")
    parser.add_argument("--sweep", action="store_true", help="grid over chunk sizes and overlaps")
    parser.add_argument("--ablate", action="store_true", help="isolate each chunking and retrieval component")
    parser.add_argument("--diff", nargs=2, metavar=("RUN_A", "RUN_B"), help="per-question flips: file.json#config")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE, help="question set (default: dev)")
    parser.add_argument("--embed", help="provider:model for every config, e.g. openai:text-embedding-3-large")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument("--save", action="store_true", help="write JSON to eval/results/")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="check every question is reachable in the corpus, then exit (no embedding)",
    )
    parser.add_argument("--strict", action="store_true", help="with --validate: lint issues also fail")
    parser.add_argument("--only", help="with --ablate: comma-separated variant names to run")
    parser.add_argument("--depth", type=int, default=0, help="for misses, show first relevant rank in each leg's top N")
    args = parser.parse_args(argv)
    only = {name.strip() for name in args.only.split(",")} if args.only else None
    if only:
        unknown = only - set(CHUNK_ABLATIONS) - set(RETRIEVAL_ABLATIONS)
        if unknown:
            parser.error(f"unknown variant(s) {sorted(unknown)}; have {sorted(CHUNK_ABLATIONS) + sorted(RETRIEVAL_ABLATIONS)}")

    if args.diff:
        return diff(*args.diff)

    questions = load_questions(args.questions)
    split = split_name(args.questions)
    docs = load_directory(args.data_dir)
    print(f"{len(questions)} {split} questions | {len(docs)} loaded section(s) | k={args.k}")

    if args.validate:
        return 1 if validate(docs, questions, strict=args.strict) else 0

    embed = embedding_name(*args.embed.split(":", 1)) if args.embed else embedding_name()
    config.require_embed_key(embed[0])
    meta = provenance(args.k, embed, args.data_dir, args.questions)

    if args.ablate:
        results = run_ablations(docs, questions, args.k, embed, only)
        if args.save:
            save("ablate", split, results, meta)
        return 0

    if args.sweep:
        results = []
        for size, overlap in [(500, 75), (800, 120), (1000, 150), (1500, 225)]:
            cfg = dict(CONFIGS["tuned"], chunk_size=size, chunk_overlap=overlap, label=f"tuned, chunk={size}/{overlap}")
            results.append(run_config(f"cs{size}", cfg, docs, questions, args.k, embed, args.depth))
    elif args.config:
        results = [run_config(args.config, CONFIGS[args.config], docs, questions, args.k, embed, args.depth)]
    else:
        results = [
            run_config(name, CONFIGS[name], docs, questions, args.k, embed, args.depth) for name in ("baseline", "tuned")
        ]

    print_comparison(results, args.k)
    if args.save:
        save("sweep" if args.sweep else "eval", split, results, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
