"""BM25 vs dense retrieval on the Devanagari electoral rolls.

There are no qrels for this corpus, so this is a capability diagnostic, not a
metrics table. Queries are grouped by what they are designed to expose:

  lexical      words that appear verbatim in the scans — BM25's home ground
  paraphrase   Hindi wording the document never uses — needs semantics
  crosslingual English questions against Devanagari pages — BM25 cannot match
  unanswerable nothing in the corpus answers these — both should stay quiet
"""

import os
import sys
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.cache import ExtractionCache            # noqa: E402
from ingestion.chunking import FixedSizeChunker         # noqa: E402
from ingestion.pdf_extractor import PDFExtractor        # noqa: E402
from ingestion.pipeline import chunk_corpus             # noqa: E402
from retrieval.bm25 import BM25Index                    # noqa: E402

ELECTORAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "fiqa", "electoral",
)

QUERIES: list[tuple[str, str, str]] = [
    ("lexical", "मतदान केंद्र", "polling station (verbatim in doc)"),
    ("lexical", "निर्वाचक नामावली", "electoral roll (verbatim)"),
    ("lexical", "कुम्हरार", "constituency name (verbatim)"),
    ("paraphrase", "वोटिंग बूथ कहाँ है", "'voting booth' — doc says मतदान केंद्र"),
    ("paraphrase", "वोटर लिस्ट", "'voter list' — doc says निर्वाचक नामावली"),
    ("paraphrase", "चुनाव क्षेत्र का नाम क्या है", "which constituency?"),
    ("crosslingual", "Where is the polling station?", "English -> Devanagari"),
    ("crosslingual", "electoral roll 2025 Bihar", "English -> Devanagari"),
    ("crosslingual", "list of voters and their names", "English -> Devanagari"),
    ("unanswerable", "How do I bake sourdough bread?", "not in corpus"),
    ("unanswerable", "रोटी कैसे बनाएं", "not in corpus (Hindi)"),
]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-pdfs", action="store_true")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args(argv)

    pdfs = sorted(os.path.join(ELECTORAL_DIR, f)
                  for f in os.listdir(ELECTORAL_DIR) if f.lower().endswith(".pdf"))
    if not args.all_pdfs:
        pdfs = pdfs[:1]

    cache = ExtractionCache()
    extractor = PDFExtractor(ocr_lang="hi", cache=cache)
    documents = []
    for path in pdfs:
        documents.extend(extractor.extract(path, clean=True))
    print(f"{len(documents)} pages from {len(pdfs)} PDF(s) | {cache.stats.summary()}")

    chunks = list(chunk_corpus(
        documents, FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)))
    print(f"{len(chunks)} chunks\n")

    from qdrant_client import QdrantClient
    from retrieval.dense import DenseIndex
    from retrieval.embedder import Embedder, MULTILINGUAL_MODEL

    bm25 = BM25Index().build(chunks)

    t0 = time.perf_counter()
    dense = DenseIndex("electoral_cmp", Embedder(MULTILINGUAL_MODEL),
                       client=QdrantClient(":memory:"))
    dense.recreate()
    dense.index(chunks)
    print(f"dense index built in {time.perf_counter() - t0:.1f}s "
          f"({MULTILINGUAL_MODEL})\n")

    stats: dict[str, dict[str, list]] = {}
    print("=" * 96)
    for group, query, note in QUERIES:
        t = time.perf_counter(); b_hits = bm25.search(query, limit=args.top)
        b_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); d_hits = dense.search(query, limit=args.top)
        d_ms = (time.perf_counter() - t) * 1000

        bucket = stats.setdefault(group, {"bm25_hit": [], "dense_hit": [],
                                          "overlap": [], "b_ms": [], "d_ms": []})
        bucket["bm25_hit"].append(bool(b_hits))
        bucket["dense_hit"].append(bool(d_hits))
        bucket["overlap"].append(jaccard([h.chunk_id for h in b_hits],
                                         [h.chunk_id for h in d_hits]))
        bucket["b_ms"].append(b_ms); bucket["d_ms"].append(d_ms)

        print(f"\n[{group}] {query}   ({note})")
        for label, hits in (("BM25 ", b_hits), ("DENSE", d_hits)):
            if not hits:
                print(f"   {label}  (no results)")
                continue
            for h in hits[:args.top]:
                page = h.chunk.page if h.chunk else "?"
                print(f"   {label}  p{page:<3} {h.score:7.3f}  "
                      f"{' '.join(h.text.split())[:88]}")

    print("\n" + "=" * 96)
    print(f"\n{'group':14s} {'BM25 returns':>13s} {'DENSE returns':>14s} "
          f"{'top-k overlap':>14s} {'BM25 ms':>9s} {'DENSE ms':>9s}")
    for group, v in stats.items():
        n = len(v["bm25_hit"])
        print(f"{group:14s} {sum(v['bm25_hit'])}/{n:<11d} {sum(v['dense_hit'])}/{n:<12d} "
              f"{sum(v['overlap'])/n:>13.2f} {sum(v['b_ms'])/n:>9.1f} "
              f"{sum(v['d_ms'])/n:>9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
