"""FiQA retrieval and selective-retrieval benchmark.

    ragenv311\\Scripts\\python eval/benchmark_fiqa.py \
        --out docs/dissertation/data/fiqa_benchmark.json

For each contamination level it runs four retrievers built from the project's
own components over the FiQA test queries (648) and dev queries (500):

    bm25           Okapi BM25 over chunks (retrieval.bm25)
    dense          multilingual-e5-small, the deployed embedder
    hybrid         Reciprocal Rank Fusion of the two (k = 60)
    hybrid_rerank  hybrid with its top 15 re-scored by the cross-encoder,
                   the production candidate depth

Contamination levels are nested: the corpus at each level is every judged
document (test and dev) plus the first n unjudged documents of one seeded
shuffle, so a higher level only adds distractors. rho = 1 - judged / total.

Per query the output keeps document-level metrics for every variant and
several confidence signals (top BM25, dense, RRF and reranker scores). Means,
confidence intervals, significance tests and risk-coverage curves are
computed from that file by eval/analyse_fiqa.py, without re-running models.
Test queries are for reporting; dev queries exist to calibrate thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import (  # noqa: E402
    _document_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ingestion.chunking import FixedSizeChunker  # noqa: E402
from ingestion.loaders import load_beir_corpus, load_beir_queries, load_qrels  # noqa: E402
from rerank.cross_encoder import DEFAULT_MODEL, get_reranker  # noqa: E402
from retrieval.bm25 import BM25Index  # noqa: E402
from retrieval.embedder import MULTILINGUAL_MODEL, get_embedder  # noqa: E402
from retrieval.local_dense import LocalDenseIndex  # noqa: E402
from retrieval.rrf import reciprocal_rank_fusion  # noqa: E402
from retrieval.types import ScoredChunk  # noqa: E402

DATA = ROOT / "data" / "fiqa"
VARIANTS = ("bm25", "dense", "hybrid", "hybrid_rerank")
CANDIDATES = 100   # chunks per retriever before fusion
RERANK_DEPTH = 15  # production IntentRouter candidate_limit


def load_data(seed: int):
    test = load_qrels(str(DATA / "qrels" / "test.tsv"))
    dev = load_qrels(str(DATA / "qrels" / "dev.tsv"))
    judged = {doc for qrels in (test, dev) for rel in qrels.values() for doc in rel}
    documents = list(load_beir_corpus(str(DATA / "corpus.jsonl")))
    judged_docs = sorted((d for d in documents if d.doc_id in judged), key=lambda d: d.doc_id)
    distractors = [d for d in documents if d.doc_id not in judged]
    random.Random(seed).shuffle(distractors)
    return judged_docs, distractors, test, dev, load_beir_queries()


def query_metrics(results: list[ScoredChunk], relevance: dict[str, int]) -> dict[str, float]:
    ranking = _document_ranking(results)
    relevant = {d for d, s in relevance.items() if s > 0}
    return {
        "ndcg10": ndcg_at_k(ranking, relevance, 10),
        "r10": recall_at_k(ranking, relevant, 10),
        "r100": recall_at_k(ranking, relevant, 100),
        "mrr10": reciprocal_rank(ranking, relevant, 10),
        "p5": precision_at_k(ranking, relevant, 5),
        "hit5": float(any(d in relevant for d in ranking[:5])),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-queries", type=int, default=None, help="per split, for smoke runs")
    parser.add_argument("--out", default=str(ROOT / "results" / "fiqa_benchmark.json"))
    args = parser.parse_args(argv)

    started_all = time.perf_counter()
    judged_docs, distractors, test_qrels, dev_qrels, queries = load_data(args.seed)
    n_judged = len(judged_docs)
    levels = [(rho, round(n_judged * rho / (1 - rho))) for rho in sorted(args.rhos)]
    max_distractors = max(n for _, n in levels)
    print(f"[*] judged docs {n_judged}; levels {levels}", flush=True)

    # Chunk judged docs then distractors in shuffle order, recording where each
    # document's chunks end, so every level is a prefix of one chunk list.
    chunker = FixedSizeChunker(chunk_size=200, overlap=0)
    chunks, boundary = [], [0]
    for doc in judged_docs + distractors[:max_distractors]:
        chunks.extend(chunker.chunk(doc))
        boundary.append(len(chunks))

    embedder = get_embedder(MULTILINGUAL_MODEL)
    index = LocalDenseIndex(f"fiqa_eval_{MULTILINGUAL_MODEL.split('/')[-1]}", embedder)
    need = [c.chunk_id for c in chunks]
    have = index.chunk_ids
    if have != need[:len(have)]:
        index.recreate()
        have = []
    if len(have) < len(need):
        print(f"[*] embedding {len(need) - len(have)} chunks ({len(have)} reused)...", flush=True)
        t = time.perf_counter()
        index.index(chunks[len(have):], batch_size=512)
        embed_seconds = time.perf_counter() - t
        print(f"[+] embedded in {embed_seconds:.0f}s", flush=True)
    else:
        embed_seconds = 0.0
        print(f"[+] reusing {len(have)} stored embeddings", flush=True)
    matrix = index.matrix

    splits = {}
    for name, qrels in (("test", test_qrels), ("dev", dev_qrels)):
        qids = [q for q in qrels if q in queries][: args.max_queries]
        splits[name] = qids
    all_qids = splits["test"] + splits["dev"]
    query_vectors = dict(zip(all_qids, embedder.embed_queries([queries[q] for q in all_qids])))
    # Per-query embedding cost, measured one query at a time as the app does.
    t = time.perf_counter()
    for q in splits["test"][:50]:
        embedder.embed_query(queries[q])
    query_embed_ms = (time.perf_counter() - t) * 1000 / max(1, min(50, len(splits["test"])))

    reranker = get_reranker(DEFAULT_MODEL)
    rerank_cache: dict[tuple[str, str], float] = {}
    output = {
        "config": {
            "embedder": MULTILINGUAL_MODEL, "reranker": DEFAULT_MODEL,
            "reranker_max_length": reranker.max_length, "chunker": "fixed-200-o0",
            "candidates_per_retriever": CANDIDATES, "rerank_depth": RERANK_DEPTH,
            "rrf_k": 60, "seed": args.seed, "judged_docs": n_judged,
            "test_queries": len(splits["test"]), "dev_queries": len(splits["dev"]),
            "corpus_sha256_12": hashlib.sha256("\n".join(need).encode()).hexdigest()[:12],
        },
        "embed_seconds": round(embed_seconds, 1),
        "query_embed_ms": round(query_embed_ms, 1),
        "levels": [],
    }

    for rho, n_distractors in levels:
        n_chunks = boundary[n_judged + n_distractors]
        level_chunks, sub = chunks[:n_chunks], matrix[:n_chunks]
        t = time.perf_counter()
        bm25 = BM25Index().build(level_chunks)
        build_seconds = time.perf_counter() - t
        print(f"\n[*] rho={rho} docs={n_judged + n_distractors} chunks={n_chunks} "
              f"(BM25 built in {build_seconds:.0f}s)", flush=True)
        records = []
        level_started = time.perf_counter()
        for split, qids in splits.items():
            qrels = test_qrels if split == "test" else dev_qrels
            for position, qid in enumerate(qids, 1):
                text = queries[qid]
                t = time.perf_counter()
                lexical = bm25.search(text, limit=CANDIDATES)
                bm25_ms = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                scores = sub @ query_vectors[qid]
                k = min(CANDIDATES, len(scores))
                top = np.argpartition(-scores, k - 1)[:k]
                top = top[np.argsort(-scores[top], kind="stable")]
                dense = [ScoredChunk(chunk_id=level_chunks[i].chunk_id, score=float(scores[i]),
                                     rank=r, chunk=level_chunks[i]) for r, i in enumerate(top, 1)]
                dense_ms = (time.perf_counter() - t) * 1000

                hybrid = reciprocal_rank_fusion(dense, lexical, k=60, limit=CANDIDATES)

                head = hybrid[:RERANK_DEPTH]
                todo = [c for c in head if (qid, c.chunk_id) not in rerank_cache]
                t = time.perf_counter()
                if todo:
                    for scored in reranker.rerank(text, todo):
                        rerank_cache[(qid, scored.chunk_id)] = scored.score
                rerank_ms = (time.perf_counter() - t) * 1000
                head = sorted(head, key=lambda c: rerank_cache[(qid, c.chunk_id)], reverse=True)
                reranked = [ScoredChunk(chunk_id=c.chunk_id, score=rerank_cache[(qid, c.chunk_id)],
                                        rank=r, chunk=c.chunk) for r, c in enumerate(head, 1)]
                reranked += hybrid[RERANK_DEPTH:]

                relevance = qrels[qid]
                records.append({
                    "qid": qid, "split": split,
                    "m": {name: query_metrics(ranked, relevance) for name, ranked in
                          (("bm25", lexical), ("dense", dense), ("hybrid", hybrid),
                           ("hybrid_rerank", reranked))},
                    "conf": {
                        "bm25": lexical[0].score if lexical else 0.0,
                        "dense": dense[0].score if dense else 0.0,
                        "rrf": hybrid[0].score if hybrid else 0.0,
                        "rerank": reranked[0].score if reranked else 0.0,
                    },
                    "ms": {"bm25": round(bm25_ms, 2), "dense_search": round(dense_ms, 2),
                           "rerank": round(rerank_ms, 1), "rerank_pairs": len(todo)},
                })
                if position % 100 == 0 or position == len(qids):
                    done = [r for r in records if r["split"] == split]
                    ndcg = np.mean([r["m"]["hybrid_rerank"]["ndcg10"] for r in done])
                    print(f"    {split} {position}/{len(qids)}  nDCG@10 (hybrid+rerank) so far {ndcg:.3f}"
                          f"  [{time.perf_counter() - level_started:.0f}s]", flush=True)

        output["levels"].append({
            "rho": rho, "distractors": n_distractors, "docs": n_judged + n_distractors,
            "chunks": n_chunks, "bm25_build_seconds": round(build_seconds, 1),
            "seconds": round(time.perf_counter() - level_started, 1), "queries": records,
        })
        test_records = [r for r in records if r["split"] == "test"]
        for name in VARIANTS:
            print(f"    {name:14s} nDCG@10={np.mean([r['m'][name]['ndcg10'] for r in test_records]):.4f} "
                  f"R@10={np.mean([r['m'][name]['r10'] for r in test_records]):.4f}", flush=True)
        output["total_seconds"] = round(time.perf_counter() - started_all, 1)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output), encoding="utf-8")
        print(f"[+] saved {out} after rho={rho}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
