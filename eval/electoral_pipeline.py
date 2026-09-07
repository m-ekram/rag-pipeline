"""End-to-end electoral-roll pipeline: OCR -> clean -> chunk -> index -> retrieve.

    python eval/electoral_pipeline.py                  # BM25 only, no downloads
    python eval/electoral_pipeline.py --dense          # + multilingual embeddings
    python eval/electoral_pipeline.py --all-pdfs       # whole electoral corpus

The electoral rolls are pure scans, so every page costs ~6s of PaddleOCR on the
first pass and is served from the extraction cache afterwards.
"""

import os
import sys
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.cache import ExtractionCache          # noqa: E402
from ingestion.chunking import FixedSizeChunker       # noqa: E402
from ingestion.pdf_extractor import PDFExtractor      # noqa: E402
from ingestion.pipeline import chunk_corpus           # noqa: E402
from retrieval.bm25 import BM25Index                  # noqa: E402

ELECTORAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "fiqa", "electoral",
)

# Hindi and English queries against the same Devanagari corpus.
QUERIES = [
    "निर्वाचक नामावली",
    "विधानसभा क्षेत्र की संख्या",
    "कुम्हरार",
    "पटना साहिब",
    "मतदान केंद्र",
    "electoral roll 2025",
    "How do I bake sourdough bread?",   # deliberately unanswerable
]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-pdfs", action="store_true")
    parser.add_argument("--dense", action="store_true",
                        help="also build a multilingual dense index (downloads ~470MB)")
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    pdfs = sorted(
        os.path.join(ELECTORAL_DIR, f)
        for f in os.listdir(ELECTORAL_DIR) if f.lower().endswith(".pdf")
    )
    if not args.all_pdfs:
        pdfs = pdfs[:1]
    print(f"PDFs: {len(pdfs)}")

    cache = ExtractionCache(enabled=not args.no_cache)
    extractor = PDFExtractor(ocr_lang="hi", cache=cache)

    started = time.perf_counter()
    documents = []
    for path in pdfs:
        print(f"  extracting {os.path.basename(path)} ...", flush=True)
        documents.extend(extractor.extract(path, clean=True))
    extract_s = time.perf_counter() - started

    if not documents:
        print("No text extracted.", file=sys.stderr)
        return 1

    ocr_pages = sum(1 for d in documents if d.metadata.get("ocr_used"))
    cached_pages = sum(1 for d in documents if d.metadata.get("from_cache"))
    confidences = [d.metadata.get("ocr_confidence") for d in documents
                   if d.metadata.get("ocr_confidence")]

    print(f"\nExtraction: {len(documents)} pages in {extract_s:.1f}s "
          f"({extract_s / len(documents):.2f}s/page)")
    print(f"  OCR pages: {ocr_pages} | from cache: {cached_pages}")
    if confidences:
        print(f"  mean OCR confidence: {sum(confidences)/len(confidences):.4f}")
    print(f"  {cache.stats.summary()}")

    chunks = list(chunk_corpus(
        documents, FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)))
    print(f"\nChunking: {len(chunks)} chunks "
          f"(size={args.chunk_size}, overlap={args.overlap})")

    bm25 = BM25Index().build(chunks)
    print(f"BM25 index: {len(bm25)} chunks")

    dense = None
    if args.dense:
        from qdrant_client import QdrantClient
        from retrieval.dense import DenseIndex
        from retrieval.embedder import Embedder, MULTILINGUAL_MODEL

        print(f"\nBuilding dense index with {MULTILINGUAL_MODEL} ...")
        t0 = time.perf_counter()
        dense = DenseIndex("electoral", Embedder(MULTILINGUAL_MODEL),
                           client=QdrantClient(":memory:"))
        dense.recreate()
        dense.index(chunks)
        print(f"  indexed in {time.perf_counter() - t0:.1f}s")

    print("\n" + "=" * 78)
    for query in QUERIES:
        print(f"\nQUERY: {query}")
        runs = [("BM25", bm25.search(query, limit=args.top))]
        if dense is not None:
            runs.append(("DENSE", dense.search(query, limit=args.top)))
        for label, hits in runs:
            if not hits:
                print(f"  {label:6s} (no results)")
                continue
            for hit in hits:
                page = hit.chunk.page if hit.chunk else "?"
                snippet = " ".join(hit.text.split())[:110]
                print(f"  {label:6s} p{page:<4} score={hit.score:7.3f}  {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
