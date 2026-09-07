"""Evaluation of dense retrieval and dense + reranker on FiQA.

Evaluates:

1. Dense retrieval baseline
2. Dense + CrossEncoder reranker

Evaluation is performed against document-level FiQA qrels while retrieval
operates over chunks. Retrieved chunk IDs are mapped back to their parent
document IDs before calculating metrics.

The evaluator does not build indexes. It only loads the existing Qdrant
collection and runs the configured retrieval pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import defaultdict
from typing import Iterable

from ingestion.loaders import load_beir_queries, load_qrels
from retrieval.types import ScoredChunk

DEFAULT_KS = (1, 3, 5, 10, 20, 50, 100)

logger = logging.getLogger(__name__)


# ============================================================
# Ranking conversion
# ============================================================

def _document_ranking(
    results: Iterable[ScoredChunk],
) -> list[str]:
    """Convert chunk-level results into a document-level ranking.

    Multiple chunks from the same document are collapsed. The first
    occurrence is retained because the input is already ranked.
    """

    seen: set[str] = set()
    ranking: list[str] = []

    for result in results:
        if result.chunk is None:
            continue

        doc_id = result.chunk.doc_id

        if doc_id in seen:
            continue

        seen.add(doc_id)
        ranking.append(doc_id)

    return ranking


# ============================================================
# Metrics
# ============================================================

def precision_at_k(
    ranking: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Calculate Precision@k.

    Divides by `k`, not by `len(ranking[:k])`: a retriever that returns only 3
    documents for k=10 has not achieved perfect precision, and normalising by
    the returned count would hide that. This matters here because the
    document-level ranking collapses chunks, so short rankings are common.
    """

    if k <= 0:
        return 0.0

    retrieved = ranking[:k]

    return len([doc_id for doc_id in retrieved if doc_id in relevant]) / k


def recall_at_k(
    ranking: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Calculate Recall@k."""

    if not relevant:
        return 0.0

    retrieved = set(ranking[:k])

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    ranking: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Calculate reciprocal rank up to k."""

    for rank, doc_id in enumerate(ranking[:k], 1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    ranking: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    """Calculate nDCG@k using FiQA relevance grades."""

    def dcg(items: list[str]) -> float:
        total = 0.0

        for rank, doc_id in enumerate(items, 1):
            gain = relevance.get(doc_id, 0)

            if gain <= 0:
                continue

            total += gain / math.log2(rank + 1)

        return total

    actual = dcg(ranking[:k])

    ideal = sorted(
        relevance.values(),
        reverse=True,
    )[:k]

    if not ideal:
        return 0.0

    ideal_dcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(ideal, 1)
        if gain > 0
    )

    if ideal_dcg == 0:
        return 0.0

    return actual / ideal_dcg


# ============================================================
# Result evaluation
# ============================================================

def evaluate_results(
    results_by_query: dict[str, Iterable[ScoredChunk]],
    qrels: dict[str, dict[str, int]],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, float]:
    """Evaluate one retrieval configuration."""

    metric_values: dict[str, list[float]] = defaultdict(list)

    for query_id, results in results_by_query.items():

        if query_id not in qrels:
            continue

        relevance = qrels[query_id]

        relevant = {
            doc_id
            for doc_id, score in relevance.items()
            if score > 0
        }

        ranking = _document_ranking(results)

        for k in ks:

            metric_values[f"Precision@{k}"].append(
                precision_at_k(
                    ranking,
                    relevant,
                    k,
                )
            )

            metric_values[f"Recall@{k}"].append(
                recall_at_k(
                    ranking,
                    relevant,
                    k,
                )
            )

            metric_values[f"MRR@{k}"].append(
                reciprocal_rank(
                    ranking,
                    relevant,
                    k,
                )
            )

            metric_values[f"nDCG@{k}"].append(
                ndcg_at_k(
                    ranking,
                    relevance,
                    k,
                )
            )

    return {
        metric: (
            sum(values) / len(values)
            if values
            else 0.0
        )
        for metric, values in metric_values.items()
    }


# ============================================================
# Retriever evaluation
# ============================================================

def evaluate_retriever(
    retriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    retrieval_limit: int = 100,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, float]:
    """Run a retriever over the FiQA queries and evaluate it.

    retrieval_limit controls how many chunks are retrieved before
    converting the results to document-level rankings.
    """

    results: dict[str, list[ScoredChunk]] = {}

    eval_query_ids = [
        query_id
        for query_id in qrels
        if query_id in queries
    ]

    total = len(eval_query_ids)

    for index, query_id in enumerate(
        eval_query_ids,
        1,
    ):
        query = queries[query_id]

        results[query_id] = retriever.retrieve(
            query,
            limit=retrieval_limit,
        )

        if index % 50 == 0 or index == total:
            print(
                f"Evaluated {index}/{total} queries...",
                flush=True,
            )

    return evaluate_results(
        results,
        qrels,
        ks=ks,
    )


# ============================================================
# Printing
# ============================================================

def print_metrics(
    name: str,
    metrics: dict[str, float],
) -> None:
    """Pretty-print evaluation metrics."""

    print()
    print(name)
    print("-" * len(name))

    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:.4f}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate dense retrieval and "
            "dense + cross-encoder reranking on FiQA."
        )
    )

    parser.add_argument(
        "--collection",
        default="smoke_test",
        help="Existing Qdrant collection name.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help=(
            "Number of chunks retrieved per query before "
            "document-level evaluation."
        ),
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=50,
        help=(
            "Number of dense candidates passed to the "
            "cross-encoder reranker."
        ),
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Evaluate the dense + CrossEncoder reranker variant."
        ),
    )

    parser.add_argument(
        "--output",
        default="results/retrieval_baseline.json",
        help="Output JSON path.",
    )

    return parser.parse_args()


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("FiQA DENSE RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    print()
    print("Loading FiQA queries and qrels...")

    queries = load_beir_queries()
    qrels = load_qrels()

    eval_query_ids = [
        query_id
        for query_id in qrels
        if query_id in queries
    ]

    print(f"Total queries available : {len(queries)}")
    print(f"Queries with qrels      : {len(eval_query_ids)}")
    print(f"Top-K retrieval limit   : {args.top_k}")
    print(f"Candidate limit         : {args.candidate_limit}")
    print(f"Reranking enabled       : {args.rerank}")

    # --------------------------------------------------------
    # Load existing Qdrant collection
    # --------------------------------------------------------

    print()
    print(
        f"Connecting to existing Qdrant collection: "
        f"{args.collection}"
    )

    from retrieval.dense import DenseIndex

    try:

        dense = DenseIndex(args.collection)

        print("Qdrant connection ready.")

    except Exception as exc:

        raise RuntimeError(
            f"Could not initialize dense retrieval: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Construct retrieval variants
    # --------------------------------------------------------

    from retrieval.hybrid import HybridRetriever

    retrievers = {}

    reranker_model_name = None

    # --------------------------------------------------------
    # Dense baseline
    # --------------------------------------------------------

    retrievers["dense"] = HybridRetriever(
        dense=dense,
        candidate_limit=args.candidate_limit,
    )

    # --------------------------------------------------------
    # Dense + reranker
    # --------------------------------------------------------

    if args.rerank:

        print()
        print("Loading cross-encoder reranker...")

        from retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker_model_name = reranker.model_name

        retrievers["dense_rerank"] = HybridRetriever(
            dense=dense,
            candidate_limit=args.candidate_limit,
            reranker=reranker,
        )

        print(
            f"Reranker model: {reranker.model_name}"
        )

    # --------------------------------------------------------
    # Check available variants
    # --------------------------------------------------------

    print()
    print("Retrieval variants:")

    for name in retrievers:
        print(f"  - {name}")

    # --------------------------------------------------------
    # Evaluation configuration
    # --------------------------------------------------------

    results = {
        "config": {
            "collection": args.collection,
            "top_k": args.top_k,
            "candidate_limit": args.candidate_limit,
            "rerank": args.rerank,
            "reranker_model": reranker_model_name,
        },
        "queries": len(eval_query_ids),
        "metrics": {},
    }

    # --------------------------------------------------------
    # Evaluate each retrieval variant
    # --------------------------------------------------------

    for name, retriever in retrievers.items():

        print()
        print("=" * 70)
        print(f"Evaluating: {name}")
        print("=" * 70)

        started = time.perf_counter()

        metrics = evaluate_retriever(
            retriever=retriever,
            queries=queries,
            qrels=qrels,
            retrieval_limit=args.top_k,
            ks=DEFAULT_KS,
        )

        elapsed = time.perf_counter() - started

        results["metrics"][name] = {
            **metrics,
            "seconds": round(elapsed, 2),
        }

        print_metrics(
            name,
            metrics,
        )

        print(
            f"Time: {elapsed:.2f}s"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_path = args.output

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            results,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved results to: {output_path}"
    )

    print()
    print("Variants evaluated:")

    for name in results["metrics"]:

        print(
            f"  {name}: "
            f"{results['metrics'][name]['seconds']:.2f}s"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()