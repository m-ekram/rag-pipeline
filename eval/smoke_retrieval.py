"""Phase 1 checkpoint: run hand-picked queries against each retrieval path and
eyeball the results side by side.

    python -m ingestion.build_index --in-domain-distractors 1706 --collection rag_rho50
    python eval/smoke_retrieval.py --collection rag_rho50

`--skip-dense` runs the BM25 half alone, which needs neither Qdrant nor the
embedding model.
"""

import os
import sys
import argparse
import logging
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.build_index import INDEX_DIR  # noqa: E402
from retrieval.types import ScoredChunk  # noqa: E402

logger = logging.getLogger(__name__)

# Ten hand-picked FiQA-flavoured questions. The last two are deliberately
# unanswerable by a finance corpus — they are the abstention cases the project's
# goal is built around, and they should look obviously bad here.
SMOKE_QUERIES = [
    "What is considered a business expense on a business trip?",
    "Can I deduct mortgage interest on a second home?",
    "How does a Roth IRA differ from a traditional IRA?",
    "What happens to my 401k when I change jobs?",
    "Is it better to lease or buy a car?",
    "How do capital gains taxes work on stock sales?",
    "What is an ETF expense ratio?",
    "Should I pay off student loans or invest?",
    "What is the average airspeed velocity of an unladen swallow?",
    "How do I treat a sourdough starter that will not rise?",
]


def show(label: str, hits: list[ScoredChunk], top: int) -> None:
    print(f"\n  --- {label} ---")
    if not hits:
        print("      (no results)")
        return
    for hit in hits[:top]:
        snippet = textwrap.shorten(hit.text.replace("\n", " "), width=150, placeholder=" ...")
        citation = hit.chunk.citation() if hit.chunk else ""
        print(f"      {hit.rank}. score={hit.score:.4f} {citation}")
        print(f"         {snippet}")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default="rag_rho50")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--queries", nargs="*", default=None)
    args = parser.parse_args(argv)

    queries = args.queries or SMOKE_QUERIES

    from retrieval.bm25 import BM25Index

    bm25_path = os.path.join(INDEX_DIR, f"{args.collection}.bm25.pkl")
    if not os.path.exists(bm25_path):
        print(f"No BM25 index at {bm25_path}. Run ingestion.build_index first.",
              file=sys.stderr)
        return 1
    bm25 = BM25Index.load(bm25_path)
    print(f"Loaded BM25 index: {len(bm25)} chunks")

    dense = None
    if not args.skip_dense:
        try:
            from retrieval.dense import DenseIndex

            dense = DenseIndex(args.collection)
            # Fail fast with a clear message rather than deep inside a query.
            if not dense.client.collection_exists(args.collection):
                print(f"Qdrant has no collection {args.collection!r}; "
                      f"skipping the dense half.", file=sys.stderr)
                dense = None
        except Exception as exc:
            print(f"Dense path unavailable ({exc}); skipping it.", file=sys.stderr)
            dense = None

    for query in queries:
        print("\n" + "=" * 78)
        print(f"QUERY: {query}")
        show("BM25", bm25.search(query, limit=args.top), args.top)
        if dense is not None:
            show("DENSE", dense.search(query, limit=args.top), args.top)
            # HybridRetriever is dense-only since commit d70c90b; the previous
            # `HybridRetriever(bm25=..., dense=...)` call raised TypeError here.
            from retrieval.hybrid import HybridRetriever

            pipeline = HybridRetriever(dense)
            show("DENSE (pipeline)", pipeline.retrieve(query, limit=args.top), args.top)

    return 0


if __name__ == "__main__":
    sys.exit(main())
