
"""
Automated contamination sweep for the FiQA retrieval pipeline.

Sweeps the number of in-domain distractor documents while keeping all
retrieval and evaluation settings fixed.

Evaluated variants:
    1. BM25
    2. Dense
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from eval.retrieval_eval import evaluate_retriever
from ingestion.loaders import load_beir_queries, load_qrels
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex
from retrieval.hybrid import HybridRetriever


# ============================================================
# Fixed experimental configuration
# ============================================================

DEFAULT_CONTAMINATION_LEVELS = (
    0,
    1706,
    5000,
    10000,
    20000,
    30000,
    40000,
    50000,
    55932,
)

DEFAULT_CHUNKER = "fixed"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 0
DEFAULT_SEED = 13

DEFAULT_CANDIDATE_LIMIT = 50
DEFAULT_TOP_K = 100

INDEX_DIR = "indexes"
RESULT_DIR = "results"


# ============================================================
# Build indexes
# ============================================================

def build_indexes(
    distractors: int,
    collection: str,
    *,
    chunker: str,
    chunk_size: int,
    overlap: int,
    seed: int,
) -> None:
    """
    Build BM25 and Dense indexes for one contamination level.

    Uses the existing ingestion.build_index pipeline so every sweep
    configuration is reproducible from its command-line arguments.
    """

    command = [
        sys.executable,
        "-m",
        "ingestion.build_index",

        "--chunker",
        chunker,

        "--chunk-size",
        str(chunk_size),

        "--overlap",
        str(overlap),

        "--in-domain-distractors",
        str(distractors),

        "--ood-distractors",
        "0",

        "--seed",
        str(seed),

        "--collection",
        collection,
    ]

    print()
    print("=" * 70)
    print(f"BUILDING INDEX: {collection}")
    print(f"In-domain distractors: {distractors}")
    print("=" * 70)

    subprocess.run(command, check=True)


# ============================================================
# Evaluate one configuration
# ============================================================

def evaluate_configuration(
    *,
    bm25_path: str,
    collection: str,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    candidate_limit: int,
    top_k: int,
) -> dict:
    """
    Load the indexes and evaluate BM25, Dense and Hybrid retrieval.
    """

    print()
    print("Loading BM25 index...")

    bm25 = BM25Index.load(bm25_path)

    print(f"BM25 chunks: {len(bm25)}")

    print(f"Connecting to Qdrant collection: {collection}")

    dense = DenseIndex(collection)

    # --------------------------------------------------------
    # BM25 wrapper — gives BM25Index a .retrieve() interface
    # compatible with evaluate_retriever.
    # --------------------------------------------------------

    class _BM25Retriever:
        """Thin wrapper so BM25Index matches the retriever protocol."""

        def __init__(self, index: BM25Index):
            self._index = index

        def retrieve(self, query: str, limit: int = 10):
            return self._index.search(query, limit=limit)

    retrievers = {
        "bm25": _BM25Retriever(bm25),

        "dense": HybridRetriever(
            dense=dense,
            candidate_limit=candidate_limit,
        ),
    }

    metrics = {}

    # --------------------------------------------------------
    # Evaluate each variant
    # --------------------------------------------------------

    for name, retriever in retrievers.items():

        print()
        print("-" * 70)
        print(f"Evaluating: {name}")
        print("-" * 70)

        started = time.perf_counter()

        result = evaluate_retriever(
            retriever=retriever,
            queries=queries,
            qrels=qrels,
            retrieval_limit=top_k,
        )

        elapsed = time.perf_counter() - started

        metrics[name] = {
            **result,
            "seconds": round(elapsed, 2),
        }

        print()
        print(json.dumps(result, indent=2))

        print(f"Time: {elapsed:.2f}s")

    return metrics


# ============================================================
# Main contamination sweep
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run the FiQA in-domain contamination sweep."
    )

    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_CONTAMINATION_LEVELS),
        help="In-domain distractor counts to evaluate.",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="Number of candidates retrieved before fusion.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Maximum retrieval depth used for evaluation.",
    )

    parser.add_argument(
        "--chunker",
        choices=("fixed", "sentence"),
        default=DEFAULT_CHUNKER,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--output",
        default="results/contamination_sweep.json",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip rebuilding an index if its BM25 index and manifest "
            "already exist."
        ),
    )

    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ========================================================
    # Experiment information
    # ========================================================

    print()
    print("=" * 70)
    print("FiQA CONTAMINATION SWEEP")
    print("=" * 70)

    print(f"Levels: {args.levels}")
    print(f"Chunker: {args.chunker}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")
    print(f"Seed: {args.seed}")
    print(f"Candidate limit: {args.candidate_limit}")
    print(f"Evaluation top-K: {args.top_k}")

    # ========================================================
    # Load evaluation data once
    # ========================================================

    print()
    print("Loading FiQA evaluation data...")

    queries = load_beir_queries()
    qrels = load_qrels()

    eval_query_ids = [
        query_id
        for query_id in qrels
        if query_id in queries
    ]

    print(f"Total queries available: {len(queries)}")
    print(f"Queries with qrels: {len(eval_query_ids)}")

    # ========================================================
    # Start sweep
    # ========================================================

    all_results = []

    sweep_started = time.perf_counter()

    for level_index, distractors in enumerate(args.levels, 1):

        collection = f"contam_{distractors}"

        bm25_path = os.path.join(
            INDEX_DIR,
            f"{collection}.bm25.pkl",
        )

        manifest_path = os.path.join(
            INDEX_DIR,
            f"{collection}.manifest.json",
        )

        print()
        print("#" * 70)

        print(
            f"SWEEP LEVEL {level_index}/{len(args.levels)}"
        )

        print(
            f"In-domain distractors: {distractors}"
        )

        print(
            f"Collection: {collection}"
        )

        print("#" * 70)

        # ====================================================
        # Build indexes
        # ====================================================

        index_exists = (
            os.path.exists(bm25_path)
            and os.path.exists(manifest_path)
        )

        if args.skip_existing and index_exists:

            print()
            print("Existing BM25 index and manifest detected.")
            print("Skipping index rebuild.")

        else:

            build_indexes(
                distractors=distractors,
                collection=collection,
                chunker=args.chunker,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                seed=args.seed,
            )

        # ====================================================
        # Load manifest
        # ====================================================

        if not os.path.exists(manifest_path):

            raise FileNotFoundError(
                f"Expected manifest not found: {manifest_path}"
            )

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as handle:

            manifest = json.load(handle)

        # ====================================================
        # Evaluate configuration
        # ====================================================

        evaluation_started = time.perf_counter()

        metrics = evaluate_configuration(
            bm25_path=bm25_path,
            collection=collection,
            queries=queries,
            qrels=qrels,
            candidate_limit=args.candidate_limit,
            top_k=args.top_k,
        )

        evaluation_seconds = (
            time.perf_counter()
            - evaluation_started
        )

        # ====================================================
        # Store results
        # ====================================================

        level_result = {
            "in_domain_distractors": distractors,

            "collection": collection,

            "manifest": manifest,

            "evaluation": {
                "candidate_limit": args.candidate_limit,

                "top_k": args.top_k,

                "queries_with_qrels": len(
                    eval_query_ids
                ),

                "seconds": round(
                    evaluation_seconds,
                    2,
                ),

                "metrics": metrics,
            },
        }

        all_results.append(level_result)

        # ====================================================
        # Save after every level
        #
        # This protects completed experiments if the computer
        # stops or the process is interrupted.
        # ====================================================

        partial_results = {
            "experiment":
                "fiqa_in_domain_contamination_sweep",

            "version": 1,

            "settings": {
                "chunker": args.chunker,

                "chunk_size": args.chunk_size,

                "overlap": args.overlap,

                "seed": args.seed,

                "candidate_limit":
                    args.candidate_limit,

                "top_k": args.top_k,

                "levels": args.levels,

                "retrievers": [
                    "bm25",
                    "dense",
                ],
            },

            "queries_available": len(queries),

            "queries_with_qrels": len(
                eval_query_ids
            ),

            "completed_levels": len(all_results),

            "results": all_results,

            "total_seconds": round(
                time.perf_counter()
                - sweep_started,
                2,
            ),
        }

        with open(
            args.output,
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                partial_results,
                handle,
                indent=2,
            )

        print()
        print(f"Progress saved to: {args.output}")

    # ========================================================
    # Final summary
    # ========================================================

    total_seconds = (
        time.perf_counter()
        - sweep_started
    )

    print()
    print("=" * 70)
    print("CONTAMINATION SWEEP COMPLETE")
    print("=" * 70)

    print(
        f"Completed levels: "
        f"{len(all_results)}/{len(args.levels)}"
    )

    print(
        f"Total time: {total_seconds:.2f}s"
    )

    print()
    print(f"Results saved to: {args.output}")

    # ========================================================
    # Recall@10 summary
    # ========================================================

    print()
    print("Recall@10 SUMMARY")
    print("-" * 70)

    for result in all_results:

        distractors = result[
            "in_domain_distractors"
        ]

        metrics = result[
            "evaluation"
        ]["metrics"]

        bm25_score = metrics[
            "bm25"
        ].get("Recall@10", 0.0)

        dense_score = metrics[
            "dense"
        ].get("Recall@10", 0.0)

        print(
            f"{distractors:6d} distractors | "
            f"BM25={bm25_score:.4f} | "
            f"Dense={dense_score:.4f}"
        )


if __name__ == "__main__":
    main()
