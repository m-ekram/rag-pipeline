import os, sys, time
from pathlib import Path
from ask import load_file, sanitize_collection_name, build_pipeline


def main() -> None:

    BASE_DIR = "data/183"
    PREFIX = "2025-EROLLGEN-S04|183-SIR-FinalRoll-Revision1-HIN-".replace("|", "-")
    SUFFIX = "-WI.pdf"

    print("=======================================================")
    print("[+] Starting Full Caching + Indexing for Booths 1 to 60")
    print("======================================================")

    start_all = time.time()
    indexed_count = 0

    for i in range(1, 61):
        fname = PREFIX + str(i) + SUFFIX
        pdf_path = Path(os.path.join(BASE_DIR, fname))
    
        if not pdf_path.exists():
            print("[!] Booth not found, skipping: " + fname)
            continue
    
        t0 = time.time()
        print("\n>>> [i" + "/60] Indexing Booth " + str(i) + ": " + fname)
    
        try:
            docs = load_file(pdf_path, ocr_lang="hin+eng", workers=1)
            col_name = sanitize_collection_name(pdf_path)
            pipeline = build_pipeline(
                docs,
                col_name,
                use_reranker=False,
                reindex=True
            )
            elapsed = time.time() - t0
            print("  [+] Booth " + str(i) + " indexed in " + str(round(elapsed, 1)) + "s")
            indexed_count += 1
        except Exception as e:
            print("  [!] Error on Booth " + str(i) + ": " + str(e))

    total_min = (time.time() - start_all) / 60
    print("\n======================================================")
    print("[+] COMPLETE: " + str(indexed_count) + "/60 Booths Indexed in " + str(round(total_min, 2)) + " mins!")
    print("======================================================")


if __name__ == "__main__":
    main()
