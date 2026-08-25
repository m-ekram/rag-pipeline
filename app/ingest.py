"""Build the vector index from everything in DATA_DIR.

    python -m app.ingest
    python -m app.ingest --rebuild --chunk-size 800 --chunk-overlap 120
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import config
from app.chunking import stats, structured_split
from app.loaders import load_directory
from app.store import build_index, save_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the FAISS index.")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--index-dir", default=config.INDEX_DIR)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--rebuild", action="store_true", help="delete any existing index first")
    parser.add_argument("--dry-run", action="store_true", help="load and chunk, but do not embed")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.dry_run:  # a dry run never embeds, so it needs no key
        config.require_embed_key()

    index_path = Path(args.index_dir)
    if args.rebuild and index_path.exists():
        shutil.rmtree(index_path)
        print(f"Removed existing index at {index_path}")

    started = time.time()

    docs = load_directory(args.data_dir)
    chunks = structured_split(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    s = stats(chunks)
    print(
        f"\n{s['count']} chunks from {s['sources']} document(s) | "
        f"chars min={s['min_chars']} median={s['median_chars']} mean={s['mean_chars']} max={s['max_chars']}"
    )

    if args.dry_run:
        print("\n--dry-run: stopping before embedding.")
        return 0

    print(f"Embedding with {config.EMBEDDING_MODEL} ({config.PROVIDER})...")
    store = build_index(chunks)
    path = save_index(store, chunks, args.index_dir)

    print(f"\nIndex written to {path.resolve()} in {time.time() - started:.1f}s")
    print("Start the API with:  uvicorn app.api:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
