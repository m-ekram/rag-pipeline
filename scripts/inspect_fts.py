import sys
from storage.fts import SQLiteFTS


def main() -> None:

    print("=== Command 1: Inspecting Table 26 (Road with width 87.17) ===")
    fts = SQLiteFTS("data/db/fts.db")
    res = fts.search("87.17", limit=3)
    for r in res:
        print(f"=== Page {r.page} ===")
        for line in r.parent_context.split("\n"):
            if "87.17" in line or "Bailey" in line or "Table 26" in line:
                print(line)

    print("\n=== Command 2: Inspecting Table 28 (Peak Hour Traffic Volume) ===")
    # Search specifically for Table 28 / Traffic Volume chunks
    res2 = fts.search("Peak hour Traffic Volume Table 28", limit=3)
    for r in res2:
        print(f"=== Page {r.page} ===")
        print(r.parent_context[:600])


if __name__ == "__main__":
    main()
