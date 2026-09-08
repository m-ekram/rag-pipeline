import argparse
import os
import sys
import time
from pathlib import Path
from qdrant_client import QdrantClient
from ask import load_file, sanitize_collection_name, build_pipeline

BASE_DIR = Path("data/183")
PREFIX = "2025-EROLLGEN-S04-183-SIR-FinalRoll-Revision1-HIN-"
SUFFIX = "-WI.pdf"


def main():
    parser = argparse.ArgumentParser(description="Resilient batch indexing for Electoral Booths into Qdrant & Cache.")
    parser.add_argument("--start", type=int, default=61, help="Starting booth number (default: 61)")
    parser.add_argument("--end", type=int, default=260, help="Ending booth number (default: 260)")
    parser.add_argument("--force", action="store_true", help="Force re-index even if collection already exists")
    args = parser.parse_args()

    qc = QdrantClient("http://localhost:6333")
    try:
        existing_cols = set(c.name for c in qc.get_collections().collections)
    except Exception as e:
        print(f"[!] Warning: Could not connect to Qdrant: {e}", flush=True)
        existing_cols = set()

    total_target = args.end - args.start + 1
    print("=" * 68, flush=True)
    print(f"[+] Starting Resilient Batch Indexing: Booths {args.start} to {args.end} ({total_target} total)", flush=True)
    print("=" * 68, flush=True)

    start_all = time.time()
    indexed_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, b in enumerate(range(args.start, args.end + 1), 1):
        fname = f"{PREFIX}{b}{SUFFIX}"
        pdf_path = BASE_DIR / fname

        if not pdf_path.exists():
            print(f"\n[{idx}/{total_target}] [!] File not found, skipping: {fname}", flush=True)
            skipped_count += 1
            continue

        col_name = sanitize_collection_name(pdf_path)

        # Skip if already in Qdrant and not forced
        if not args.force and col_name in existing_cols:
            try:
                cnt = qc.count(col_name).count
                if cnt > 0:
                    print(f"[{idx}/{total_target}] [=] Booth {b} ALREADY indexed ({cnt} vectors in '{col_name}'). Skipping!", flush=True)
                    skipped_count += 1
                    continue
            except Exception:
                pass

        t0 = time.time()
        print(f"\n[{idx}/{total_target}] >>> Indexing Booth {b}: {fname}", flush=True)

        try:
            # Canonical ocr_lang='hin+eng' ensures cache hit with ask.py
            docs = load_file(pdf_path, ocr_lang="hin+eng", workers=1)
            pipeline = build_pipeline(
                docs,
                col_name,
                use_reranker=False,
                reindex=True,
            )
            elapsed = time.time() - t0
            indexed_count += 1

            processed = indexed_count + skipped_count
            avg_time = (time.time() - start_all) / max(1, indexed_count)
            rem_to_index = (total_target - processed)
            eta_mins = (rem_to_index * avg_time) / 60

            print(f"  [+] Booth {b} indexed in {elapsed:.1f}s | {indexed_count} indexed | ETA: ~{eta_mins:.1f} mins remaining", flush=True)
        except Exception as e:
            failed_count += 1
            print(f"  [!] ERROR on Booth {b}: {e}", flush=True)

    total_mins = (time.time() - start_all) / 60
    print("\n" + "=" * 68, flush=True)
    print(f"[+] BATCH FINISHED: {indexed_count} indexed, {skipped_count} skipped, {failed_count} errors in {total_mins:.2f} mins", flush=True)
    print("=" * 68, flush=True)


if __name__ == "__main__":
    main()
