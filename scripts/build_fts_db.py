"""Build persistent SQLite FTS database for PMP 2031 report."""

import time
from pathlib import Path

from ask import load_file
from ingestion.chunking import StructureAwareParentChildChunker
from ingestion.pipeline import chunk_corpus
from storage.fts import SQLiteFTS


def main():
    pdf_path = Path("data/pmp-2031-report.pdf")
    db_path = "data/db/fts.db"
    Path("data/db").mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading and extracting {pdf_path}...")
    t0 = time.time()
    docs = load_file(pdf_path, ocr_lang="en", workers=3)
    print(f"[+] Loaded {len(docs)} pages in {time.time() - t0:.2f}s.")

    print("[*] Chunking with StructureAwareParentChildChunker...")
    chunker = StructureAwareParentChildChunker(rows_per_child=3, text_child_words=200)
    chunks = list(chunk_corpus(docs, chunker))
    print(f"[+] Created {len(chunks)} chunks.")

    print(f"[*] Indexing chunks into {db_path}...")
    fts = SQLiteFTS(db_path)
    fts.index_chunks(chunks)
    print(f"[+] Done! Total indexed chunks in FTS: {fts.count()}")


if __name__ == "__main__":
    main()
