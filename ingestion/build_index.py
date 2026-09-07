"""Build BM25 and/or dense indexes for a configurable document corpus.

Supported sources:
    FiQA/BEIR corpus:
        python -m ingestion.build_index --source fiqa ...

    Local PDF:
        python -m ingestion.build_index \
            --source pdf \
            --pdf data/pmp-2031-report.pdf ...

The indexing layer is intentionally independent of evaluation. FiQA-specific
contamination and qrels logic stays out of the generic PDF path.
"""

import argparse
import json
import logging
import os
import sys
import time

from .chunking import FixedSizeChunker, SentenceAwareChunker
from .loaders import load_beir_corpus, load_pdf
from .pipeline import (
    build_corpus,
    chunk_corpus,
    contamination_ratio,
    judged_doc_ids,
)

logger = logging.getLogger(__name__)

INDEX_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
    "indexes",
)


def make_chunker(name: str, size: int, overlap: int):
    if name == "fixed":
        return FixedSizeChunker(
            chunk_size=size,
            overlap=overlap,
        )

    if name == "sentence":
        return SentenceAwareChunker(
            max_words=size,
        )

    raise ValueError(f"Unknown chunker: {name}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------
    # Corpus source
    # ------------------------------------------------------------

    p.add_argument(
        "--source",
        choices=("fiqa", "pdf"),
        default="fiqa",
        help="Corpus source.",
    )

    p.add_argument(
        "--pdf",
        default=None,
        help="Path to a PDF when --source pdf is used.",
    )

    p.add_argument(
        "--ocr-lang",
        default="en",
        help="OCR language passed to the PDF extractor.",
    )

    p.add_argument(
        "--per-page",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep PDF pages as separate Documents. "
            "Use --no-per-page to combine the PDF."
        ),
    )

    # ------------------------------------------------------------
    # FiQA-specific corpus controls
    # ------------------------------------------------------------

    p.add_argument(
        "--in-domain-distractors",
        type=int,
        default=0,
        help=(
            "FiQA only: unjudged FiQA docs to add. "
            "-1 means all of them."
        ),
    )

    p.add_argument(
        "--ood-distractors",
        type=int,
        default=0,
        help="FiQA only: documents from the noisy corpus.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for FiQA corpus sampling.",
    )

    # ------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------

    p.add_argument(
        "--chunker",
        choices=("fixed", "sentence"),
        default="fixed",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=200,
    )

    p.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Fixed chunker only.",
    )

    # ------------------------------------------------------------
    # Index configuration
    # ------------------------------------------------------------

    p.add_argument(
        "--collection",
        default="rag_chunks",
        help="Qdrant collection name.",
    )

    p.add_argument(
        "--skip-dense",
        action="store_true",
        help="Build BM25 only.",
    )

    p.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Build dense index only.",
    )

    p.add_argument(
        "--out-dir",
        default=INDEX_DIR,
    )

    return p.parse_args(argv)


def load_documents(args):
    """Load Documents according to the selected corpus source."""

    if args.source == "fiqa":

        in_domain = (
            None
            if args.in_domain_distractors < 0
            else args.in_domain_distractors
        )

        documents = build_corpus(
            in_domain_distractors=in_domain,
            out_of_domain_distractors=args.ood_distractors,
            seed=args.seed,
        )

        judged = judged_doc_ids()

        return documents, {
            "source": "fiqa",
            "contamination": contamination_ratio(
                documents,
                judged,
            ),
            "in_domain_distractors": args.in_domain_distractors,
            "ood_distractors": args.ood_distractors,
            "seed": args.seed,
        }

    # ------------------------------------------------------------
    # Generic PDF source
    # ------------------------------------------------------------

    if args.source == "pdf":

        if not args.pdf:
            raise ValueError(
                "--pdf is required when --source pdf is used"
            )

        if not os.path.exists(args.pdf):
            raise FileNotFoundError(
                f"PDF not found: {args.pdf}"
            )

        documents = list(
            load_pdf(
                args.pdf,
                clean=True,
                per_page=args.per_page,
                ocr_lang=args.ocr_lang,
            )
        )

        return documents, {
            "source": "pdf",
            "pdf_path": os.path.abspath(args.pdf),
            "ocr_lang": args.ocr_lang,
            "per_page": args.per_page,
            "contamination": None,
        }

    raise ValueError(f"Unsupported source: {args.source}")


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args(argv)

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    chunker = make_chunker(
        args.chunker,
        args.chunk_size,
        args.overlap,
    )

    started = time.perf_counter()

    # ------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------

    logger.info(
        "Loading corpus source: %s",
        args.source,
    )

    documents, source_metadata = load_documents(args)

    if not documents:
        raise RuntimeError(
            "No documents were loaded."
        )

    logger.info(
        "Loaded %d documents",
        len(documents),
    )

    # ------------------------------------------------------------
    # Chunk corpus
    # ------------------------------------------------------------

    chunks = list(
        chunk_corpus(
            documents,
            chunker,
        )
    )

    if not chunks:
        raise RuntimeError(
            "No chunks were produced."
        )

    logger.info(
        "Chunked %d documents into %d chunks with %s",
        len(documents),
        len(chunks),
        chunker.name,
    )

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------

    manifest = {
        "source": args.source,
        "chunker": chunker.name,
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "collection": args.collection,
        **source_metadata,
    }

    # ------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------

    if not args.skip_bm25:

        from retrieval.bm25 import BM25Index

        bm25_path = os.path.join(
            args.out_dir,
            f"{args.collection}.bm25.pkl",
        )

        logger.info(
            "Building BM25 index..."
        )

        BM25Index().build(
            chunks
        ).save(
            bm25_path
        )

        manifest["bm25_path"] = bm25_path

        logger.info(
            "Saved BM25 index to %s",
            bm25_path,
        )

    # ------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------

    if not args.skip_dense:

        from retrieval.dense import DenseIndex

        logger.info(
            "Building dense Qdrant index..."
        )

        dense = DenseIndex(
            args.collection
        )

        dense.recreate()

        dense.index(
            chunks,
            show_progress=True,
        )

        logger.info(
            "Indexed %d chunks into Qdrant collection %r",
            len(chunks),
            args.collection,
        )

    # ------------------------------------------------------------
    # Final manifest
    # ------------------------------------------------------------

    manifest["build_seconds"] = round(
        time.perf_counter() - started,
        2,
    )

    manifest_path = os.path.join(
        args.out_dir,
        f"{args.collection}.manifest.json",
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )

    logger.info(
        "Manifest saved to %s",
        manifest_path,
    )

    logger.info(
        "Build complete: %s",
        json.dumps(
            manifest,
            indent=2,
        ),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())