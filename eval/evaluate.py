"""Measure retrieval quality, so tuning is evidence-driven rather than vibes.

    python -m eval.evaluate --compare          # baseline vs tuned, side by side
    python -m eval.evaluate --config tuned     # just the shipping config
    python -m eval.evaluate --ablate           # every variant, one at a time
    python -m eval.evaluate --sweep            # grid over chunk sizes
    python -m eval.evaluate --validate         # is every question even answerable?
    python -m eval.evaluate --compare --questions eval/questions_heldout.yaml --save
    python -m eval.evaluate --ablate --crossval --questions eval/questions.yaml eval/questions_heldout.yaml
    python -m eval.evaluate --diff eval/results/A.json#tuned eval/results/B.json#hybrid-bm25
    python -m eval.evaluate --compare --embed openai:text-embedding-3-small --save

Question sets:
  eval/questions.yaml             dev (36) - tuned against since round 1
  eval/questions_heldout.yaml     held-out v1 (50) - scored once in round 1;
                                  now part of the round-2 tuning pool
  eval/questions_heldout_v2.yaml  held-out v2 - frozen after the round-2 lock,
                                  scored once. The headline number comes from
                                  whichever held-out set is newest and untouched.

Several files can be passed at once (the "pool"); every result then carries a
per-set breakdown, so a change that wins by fitting one set is visible.

--crossval (with --ablate over exactly two sets) is the honest estimate of how
the variant *selection* generalises: pick the best variant on set A and score
it on set B, then the reverse. Scoring a chosen config on the questions it was
chosen on is optimistic - in round 1 by 10.9 points (dev 88.9% -> held-out 78.0%).

Metrics, all computed at k = TOP_K:
  hit_rate   fraction of questions where >=1 relevant chunk made the top k.
             This is the "top-5 retrieval relevance" number - if it is low, no
             prompt engineering downstream will save the answer. Reported with
             a Wilson 95% interval: on 36-60 questions one question moves the
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

`--save` writes JSON to eval/results/ with the git commit, the question sets'
hashes and the embedding model, so every number quoted in the README can be
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
        "header_target": "all",
        "hybrid": False,
        "mmr_lambda": None,
        "rerank": False,
        "doc2query": False,
        "rewrite": False,
    },
    "tuned": {
        "label": (
            f"structure-aware chunks + overlap + '{config.HEADER_MODE}' headers"
            f"{' (lexical leg only)' if config.HEADER_TARGET == 'sparse' else ''}, "
            f"{f'hybrid {config.SPARSE.upper()}/dense' if config.USE_HYBRID else 'dense'}"
            f"{', cross-encoder rerank' if config.RERANK else ''}"
            f"{', doc2query' if config.DOC2QUERY else ''}"
            f"{', query rewriting' if config.QUERY_REWRITE else ''}"
        ),
        "splitter": "structured",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "header_mode": config.HEADER_MODE,
        "header_target": config.HEADER_TARGET,
        "hybrid": config.USE_HYBRID,
        "mmr_lambda": config.MMR_LAMBDA,
        "rerank": config.RERANK,
        "doc2query": config.DOC2QUERY,
        "rewrite": config.QUERY_REWRITE,
    },
}

# Chunking ablations: different chunk text means different vectors, so each
# gets its own index.
CHUNK_ABLATIONS = {
    "headers:none": {"header_mode": "none"},
    "headers:title": {"header_mode": "title"},
    "headers:path": {"header_mode": "path"},
    "headers:path-clean": {"header_mode": "path-clean"},
    # Round 2: keep the breadcrumb for the lexical leg (and the LLM, and
    # citations) but embed only the body. On held-out v1, 5 of the tuned
    # config's 11 misses retrieved the right page but a chunk without the
    # answer: a shared header makes sibling chunks look alike to the dense leg.
    "headers:sparse-only": {"header_target": "sparse"},
    "headers:sparse-only+bge-small": {"header_target": "sparse", "embed": "local:BAAI/bge-small-en-v1.5"},
}

_BGE_RERANK = "BAAI/bge-reranker-base"
_COLBERT = "answerdotai/answerai-colbert-small-v1"
_MINILM = "Xenova/ms-marco-MiniLM-L-6-v2"

# Retrieval ablations: they share the tuned chunk set, so the embedding cost is
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
    "hybrid-bm25": {"hybrid": True, "mmr_lambda": 1.0, "rerank": False, "sparse": "bm25"},
}

# Round 2: local generation (Phi-3.5 on ONNX Runtime) aimed at paraphrase
# misses. doc2query changes what is indexed (its own index); rewriting only
# changes how the index is queried.
GENERATION_ABLATIONS = {
    "doc2query": {"doc2query": True},
    "rewrite": {"rewrite": True},
    "doc2query+rewrite": {"doc2query": True, "rewrite": True},
}

# Every variant --ablate/--only knows about. "tuned" (the shipping config, no
# overrides) comes first, so on a tie --crossval keeps the incumbent.
VARIANTS = {"tuned": {}, **CHUNK_ABLATIONS, **RETRIEVAL_ABLATIONS, **GENERATION_ABLATIONS}


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
        q["_set"] = split_name(path)
    return questions


def load_question_sets(paths: list[Path]) -> list[dict]:
    questions = [q for path in paths for q in load_questions(path)]
    seen: set[str] = set()
    for q in questions:
        if q["question"] in seen:
            raise SystemExit(f"Duplicate question across sets: {q['question']}")
        seen.add(q["question"])
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
    chunks = structured_split(
        docs,
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        add_headers=cfg["header_mode"] != "none",
        header_mode=cfg["header_mode"] if cfg["header_mode"] != "none" else None,
        header_target=cfg.get("header_target"),
    )
    if cfg.get("doc2query"):
        from app.doc2query import expand_chunks

        expand_chunks(chunks)
    return chunks


def embed_index(chunks: list[Document], embed: tuple[str, str]):
    provider, model = embed
    return build_index(chunks, show_progress=False, embeddings=get_embeddings(provider, model), model_name=model)


def make_retriever(store, chunks: list[Document], cfg: dict, k: int):
    if not cfg["hybrid"] and cfg["mmr_lambda"] is None and not cfg["rerank"] and not cfg.get("rewrite"):
        return dense_only_retriever(store, k)
    with overrides(
        USE_HYBRID=cfg["hybrid"],
        MMR_LAMBDA=1.0 if cfg["mmr_lambda"] is None else cfg["mmr_lambda"],
        RERANK=cfg["rerank"],
        QUERY_REWRITE=cfg.get("rewrite", False),
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
        legs["sparse"] = build_sparse(chunks).model_copy(update={"k": depth})
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
    by_set: dict[str, list[int]] = {}
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
        qset = question.get("_set", "dev")
        by_set.setdefault(qset, [0, 0])
        by_set[qset][0] += bool(rank)
        by_set[qset][1] += 1

        detail.append(
            {
                "question": question["question"],
                "set": qset,
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
        "by_set": {qset: f"{h}/{t}" for qset, (h, t) in by_set.items()},
        "detail": detail,
    }


def _summary_line(r: dict) -> str:
    lo, hi = r["hit_rate_ci95"]
    sets = f" | {r['by_set']}" if len(r.get("by_set", {})) > 1 else ""
    return (
        f"hit@{r['k']} {r['hit_rate']:6.1%} ({r['hits']}/{r['questions']}, 95% CI {lo:.0%}-{hi:.0%}) | "
        f"hit@1 {r['hit@1']:.0%} hit@3 {r['hit@3']:.0%} | MRR {r['mrr']:.3f} | P@{r['k']} {r['precision']:.1%} | "
        f"{r['latency_ms']['mean']:.0f} ms | {r['by_tag']}{sets}"
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
    """Run each variant on top of the shipping config.

    An index is built once per distinct (chunking, header placement, embedding)
    and shared by every variant that needs it; `only` restricts the run to the
    named variants (all when None).
    """
    tuned = CONFIGS["tuned"]
    built: dict[tuple, tuple[list[Document], object]] = {}
    results = []
    for name, spec in VARIANTS.items():
        if only is not None and name not in only:
            continue
        cfg = dict(tuned, **spec)
        emb = embedding_name(*cfg["embed"].split(":", 1)) if cfg.get("embed") else embed
        key = (
            cfg["splitter"],
            cfg["chunk_size"],
            cfg["chunk_overlap"],
            cfg["header_mode"],
            cfg.get("header_target"),
            bool(cfg.get("doc2query")),
            emb,
        )
        if key not in built:
            chunks = build_chunks(docs, cfg)
            built[key] = (chunks, embed_index(chunks, emb))
        chunks, store = built[key]
        print(f"  {name:30}", end="", flush=True)
        result = score(make_retriever(store, chunks, cfg, k), questions, k)
        print(f" {_summary_line(result)}")
        results.append(
            {
                "config": name,
                "label": json.dumps(spec, default=str),
                "embedding": f"{emb[0]}:{emb[1]}",
                "chunks": len(chunks),
                **result,
            }
        )
    return results


def _per_set(result: dict, qset: str) -> tuple[int, int, float]:
    """(hits, questions, MRR) of one question set inside a pooled result."""
    rows = [d for d in result["detail"] if d.get("set") == qset]
    hits = sum(1 for d in rows if d["hit"])
    mrr = sum(1.0 / d["first_relevant_rank"] for d in rows if d["hit"]) / len(rows) if rows else 0.0
    return hits, len(rows), mrr


def crossval(results: list[dict], sets: list[str]) -> dict:
    """2-fold selection estimate: choose on one set, score on the other.

    Selection is by hits, then MRR, on the choosing set only; ties go to the
    earlier variant (the incumbent "tuned" is listed first). The honest estimate
    pools the two out-of-fold scores - neither was seen by the choice it scores.
    """
    if len(sets) != 2:
        raise SystemExit(f"--crossval needs exactly two question sets, got {sets}")
    folds = []
    for choose_on, score_on in ((sets[0], sets[1]), (sets[1], sets[0])):
        best = max(results, key=lambda r: _per_set(r, choose_on)[0] + _per_set(r, choose_on)[2] / 10)
        hits, n, mrr = _per_set(best, score_on)
        folds.append({"chosen_on": choose_on, "chosen": best["config"], "scored_on": score_on, "hits": hits, "n": n, "mrr": mrr})
    hits = sum(f["hits"] for f in folds)
    n = sum(f["n"] for f in folds)
    lo, hi = wilson(hits, n)
    return {"folds": folds, "hits": hits, "n": n, "honest_estimate": hits / n, "ci95": [round(lo, 4), round(hi, 4)]}


def print_crossval(cv: dict) -> None:
    print("\n" + "=" * 79)
    for f in cv["folds"]:
        print(
            f"choose on {f['chosen_on']:<8} -> {f['chosen']:<30} scored on {f['scored_on']:<8}: "
            f"{f['hits']}/{f['n']} ({f['hits'] / f['n']:.1%})"
        )
    lo, hi = cv["ci95"]
    print("-" * 79)
    print(f"honest estimate (out-of-fold): {cv['hits']}/{cv['n']} = {cv['honest_estimate']:.1%}  (95% CI {lo:.0%}-{hi:.0%})")
    print("=" * 79)


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
        print(f"    + [{hits_b[q].get('set', '')}] {q}")
    print(f"  lost ({len(lost)}):")
    for q in lost:
        print(f"    - [{hits_a[q].get('set', '')}] {q}")
    better = sum(1 for _, ra, rb in moved if rb < ra)
    print(f"  rank changes among shared hits: {better} better, {len(moved) - better} worse")
    return 0


def provenance(k: int, embed: tuple[str, str], data_dir: str, question_paths: list[Path]) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "questions_file": ", ".join(p.name for p in question_paths),
        "questions_sha1": {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in question_paths},
        "embedding": f"{embed[0]}:{embed[1]}",
        "k": k,
        "data_dir": data_dir,
        "config": config.summary(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save(kind: str, split: str, results: list[dict], meta: dict, extra: dict | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{kind}-{split}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"meta": meta, "results": results, **(extra or {})}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--config", choices=sorted(CONFIGS), help="evaluate a single configuration")
    parser.add_argument("--compare", action="store_true", help="baseline vs tuned (default)")
    parser.add_argument("--sweep", action="store_true", help="grid over chunk sizes and overlaps")
    parser.add_argument("--ablate", action="store_true", help="run each variant on top of the shipping config")
    parser.add_argument("--crossval", action="store_true", help="with --ablate over two sets: out-of-fold selection estimate")
    parser.add_argument("--diff", nargs=2, metavar=("RUN_A", "RUN_B"), help="per-question flips: file.json#config")
    parser.add_argument(
        "--questions", type=Path, nargs="+", default=[QUESTIONS_FILE], help="question set file(s) (default: dev)"
    )
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
        unknown = only - set(VARIANTS)
        if unknown:
            parser.error(f"unknown variant(s) {sorted(unknown)}; have {sorted(VARIANTS)}")
    if args.crossval and not args.ablate:
        parser.error("--crossval needs --ablate (it chooses among the ablation variants)")

    if args.diff:
        return diff(*args.diff)

    questions = load_question_sets(args.questions)
    sets = list(dict.fromkeys(q["_set"] for q in questions))
    split = sets[0] if len(sets) == 1 else "pool"
    docs = load_directory(args.data_dir)
    print(f"{len(questions)} questions ({', '.join(sets)}) | {len(docs)} loaded section(s) | k={args.k}")

    if args.validate:
        return 1 if validate(docs, questions, strict=args.strict) else 0

    embed = embedding_name(*args.embed.split(":", 1)) if args.embed else embedding_name()
    config.require_embed_key(embed[0])
    meta = provenance(args.k, embed, args.data_dir, args.questions)

    if args.ablate:
        results = run_ablations(docs, questions, args.k, embed, only)
        cv = crossval(results, sets) if args.crossval else None
        if cv:
            print_crossval(cv)
        if args.save:
            save("ablate", split, results, meta, {"crossval": cv} if cv else None)
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
