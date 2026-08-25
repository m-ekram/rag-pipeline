"""Fetch the FastAPI documentation and flatten it into a clean RAG corpus.

    python scripts/prepare_fastapi_docs.py

FastAPI's docs are MkDocs-Material markdown, which carries three constructs that
hurt retrieval if fed in raw:

1. `{* ../../docs_src/body/tutorial001.py hl[2] *}` - the code examples are not
   in the markdown at all, they are include directives pointing at real .py
   files. Left as-is, every "how do I do X" chunk loses the code that answers
   it. We inline the referenced file as a fenced block.
2. `/// note` ... `///` - admonition fences. The markers are noise; the text
   inside them is often the most specific content on the page. We keep the text
   and turn the marker into a bold label.
3. `## Heading { #explicit-anchor }` - anchor suffixes that add nothing.

We also skip pages that are not documentation: the 694 KB release-notes
changelog would be ~40% of the corpus and is almost entirely "Fix typo. PR #123
by @user", which is retrieval poison.

Licence: FastAPI is MIT (Copyright (c) 2018 Sebastian Ramirez).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
DOCS_SUBDIR = "docs/en/docs"

# Not documentation: changelogs, contributor lists, link directories.
SKIP_FILES = {
    "release-notes.md",
    "fastapi-people.md",
    "external-links.md",
    "translations.md",
    "translation-banner.md",
    "newsletter.md",
    "_llm-test.md",
    "management.md",
    "management-tasks.md",
}
SKIP_DIRS = {"resources", "img", "css", "js"}

# The directive takes optional modifiers - `hl[2]` (highlight), `ln[19:21]`
# (line range), `title["app/main.py"]` - in any combination. Highlight and range
# are ignored and the whole file is inlined: the extra lines are valid context,
# and a fragment starting mid-function is worse to retrieve than a complete
# example. The title, when present, is the file's real path in a multi-module
# example, so it makes a better caption than the flat source filename.
INCLUDE_RE = re.compile(
    r"^\{\*\s*(\S+?)\s*((?:\w+\[[^\]]*\]\s*)*)\*\}\s*$",
    re.MULTILINE,
)
TITLE_RE = re.compile(r'title\["?([^"\]]+)"?\]')
# Admonitions nest, and a nested block uses one more slash: /// inside ////.
ADMONITION_OPEN_RE = re.compile(r"^/{3,}\s+(\w+)(?:\s*\|\s*(.*))?$", re.MULTILINE)
ADMONITION_CLOSE_RE = re.compile(r"^/{3,}\s*$", re.MULTILINE)
ANCHOR_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$", re.MULTILINE)


def clone(repo_dir: Path) -> None:
    """Shallow, blobless, sparse clone - a few MB instead of the full history."""
    if (repo_dir / DOCS_SUBDIR).is_dir():
        print(f"Using existing checkout at {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} -> {repo_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(repo_dir)],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", DOCS_SUBDIR, "docs_src"],
        cwd=repo_dir,
        check=True,
    )


def resolve_include(target: str, repo_dir: Path) -> Path | None:
    """`../../docs_src/body/tutorial001.py` resolves from the repo root."""
    cleaned = re.sub(r"^(\.\./)+", "", target.strip())
    path = repo_dir / cleaned
    return path if path.is_file() else None


def inline_includes(text: str, repo_dir: Path, counters: dict) -> str:
    def replace(match: re.Match) -> str:
        target = match.group(1)
        path = resolve_include(target, repo_dir)
        if path is None:
            counters["missing"] += 1
            return ""
        counters["inlined"] += 1
        lang = "python" if path.suffix == ".py" else path.suffix.lstrip(".")
        code = path.read_text(encoding="utf-8", errors="ignore").strip()
        title_match = TITLE_RE.search(match.group(2) or "")
        caption = title_match.group(1) if title_match else path.name
        return f"Example (`{caption}`):\n\n```{lang}\n{code}\n```"

    return INCLUDE_RE.sub(replace, text)


def flatten_admonitions(text: str) -> str:
    def open_marker(match: re.Match) -> str:
        kind = match.group(1).capitalize()
        title = (match.group(2) or "").strip()
        return f"**{kind}: {title}**" if title else f"**{kind}:**"

    text = ADMONITION_OPEN_RE.sub(open_marker, text)
    return ADMONITION_CLOSE_RE.sub("", text)


def clean(text: str, repo_dir: Path, counters: dict) -> str:
    text = inline_includes(text, repo_dir, counters)
    text = flatten_admonitions(text)
    text = ANCHOR_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a FastAPI docs corpus.")
    parser.add_argument("--repo", default=".cache/fastapi", help="where to clone the repo")
    parser.add_argument("--out", default="data/fastapi", help="corpus output directory")
    parser.add_argument("--clean", action="store_true", help="empty the output dir first")
    args = parser.parse_args(argv)

    repo_dir = Path(args.repo).resolve()
    out_dir = Path(args.out)

    clone(repo_dir)

    docs_root = repo_dir / DOCS_SUBDIR
    if not docs_root.is_dir():
        raise SystemExit(f"Docs not found at {docs_root}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counters = {"inlined": 0, "missing": 0}
    written = skipped = 0
    total_bytes = 0

    for path in sorted(docs_root.rglob("*.md")):
        rel = path.relative_to(docs_root)
        if path.name in SKIP_FILES or set(rel.parts[:-1]) & SKIP_DIRS:
            skipped += 1
            continue

        cleaned = clean(path.read_text(encoding="utf-8", errors="ignore"), repo_dir, counters)
        if len(cleaned) < 200:  # stubs and redirect pages
            skipped += 1
            continue

        # Flatten tutorial/body.md -> tutorial__body.md so the source metadata
        # stays readable in citations without nested directories.
        dest = out_dir / rel.as_posix().replace("/", "__")
        dest.write_text(cleaned, encoding="utf-8")
        written += 1
        total_bytes += len(cleaned)

    print(
        f"\nWrote {written} files ({total_bytes / 1_048_576:.1f} MB) to {out_dir}"
        f" | skipped {skipped}"
        f" | inlined {counters['inlined']} code examples"
        + (f" | {counters['missing']} includes unresolved" if counters["missing"] else "")
    )
    print("\nNext:  python -m app.ingest --rebuild")
    return 0


if __name__ == "__main__":
    sys.exit(main())
