"""Load-test ingest + retrieval at 10k+ pages without the compute bill.

    python -m bench.scale_test                     # 12,000 PDF pages, fake embeddings
    python -m bench.scale_test --pages 20000
    python -m bench.scale_test --no-pdf            # skip PDF generation + extraction
    python -m bench.scale_test --embedding openai  # real text-embedding-3-small (~$0.15)

What is real: multi-page PDFs (with running headers, page numbers and a
bookmark outline) extracted through app.loaders, chunking, the on-disk
embedding cache, FAISS build/save/load, BM25, and the same hybrid retriever
the API serves. What is fake by default: the vectors - deterministic
hash-seeded 1536-d vectors, the size text-embedding-3-small produces - so
memory and FAISS/BM25 behaviour match a real run while embedding costs
nothing. Retrieval *quality* is meaningless here; eval/ measures that.

Page text is sampled from the real FastAPI docs in data/, so token statistics
look like technical documentation rather than lorem ipsum.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import psutil

import config

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
_FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
PAGES_PER_CHAPTER = 25


class MemoryMonitor:
    """Samples RSS of this process plus loader workers every 50 ms."""

    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.peak = 0
        self.stage_peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def rss(self) -> int:
        total = self.proc.memory_info().rss
        for child in self.proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            value = self.rss()
            self.peak = max(self.peak, value)
            self.stage_peak = max(self.stage_peak, value)

    def __enter__(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def _mb(n: int) -> float:
    return round(n / 1_048_576, 1)


def source_paragraphs(source_dir: Path) -> list[str]:
    paragraphs = []
    for path in sorted(source_dir.rglob("*.md")):
        text = _FENCE.sub("", path.read_text(encoding="utf-8", errors="ignore"))
        for block in re.split(r"\n\s*\n", text):
            # Standard PDF fonts only cover Latin-1; drop what they cannot draw.
            block = " ".join(block.split()).encode("ascii", "ignore").decode()
            if len(block) >= 80 and not block.startswith(("#", "|", "{", "<", "!")):
                paragraphs.append(block)
    if len(paragraphs) < 50:
        raise SystemExit(
            f"Found only {len(paragraphs)} paragraphs under {source_dir} to sample page text from. "
            "Build the corpus first:  python scripts/prepare_fastapi_docs.py"
        )
    return paragraphs


def page_texts(paragraphs: list[str], pages: int, chars: int, rng: random.Random):
    for _ in range(pages):
        page, n = [], 0
        while n < chars:
            p = rng.choice(paragraphs)
            page.append(p)
            n += len(p)
        yield page


def write_pdfs(corpus_dir: Path, paragraphs: list[str], args) -> None:
    """Generate the corpus once; reruns with the same spec reuse it."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    spec = {"pages": args.pages, "per_file": args.pages_per_file, "chars": args.chars_per_page, "seed": args.seed}
    marker = corpus_dir / "SPEC.json"
    if marker.exists() and json.loads(marker.read_text()) == spec:
        print(f"Reusing generated corpus in {corpus_dir}")
        return
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True)

    started = time.perf_counter()
    width, height = letter
    pages = page_texts(paragraphs, args.pages, args.chars_per_page, random.Random(args.seed))
    remaining, file_no = args.pages, 0
    while remaining > 0:
        count = min(args.pages_per_file, remaining)
        remaining -= count
        file_no += 1
        pdf = canvas.Canvas(str(corpus_dir / f"manual-{file_no:03d}.pdf"), pagesize=letter)
        pdf.setTitle(f"Synthetic Manual {file_no}")
        for p in range(1, count + 1):
            chapter = (p - 1) // PAGES_PER_CHAPTER + 1
            if (p - 1) % PAGES_PER_CHAPTER == 0:
                pdf.bookmarkPage(f"ch{chapter}")
                pdf.addOutlineEntry(f"Chapter {chapter}", f"ch{chapter}", level=0)
            # Running header and page number: the boilerplate the loader strips.
            pdf.setFont("Helvetica", 8)
            pdf.drawString(54, height - 36, f"Synthetic Manual {file_no} - Chapter {chapter}")
            pdf.drawString(width / 2, 30, str(p))
            pdf.setFont("Helvetica", 9)
            y = height - 64
            for paragraph in next(pages):
                for line in textwrap.wrap(paragraph, 105):
                    if y < 50:
                        break
                    pdf.drawString(54, y, line)
                    y -= 11
                y -= 6
                if y < 50:
                    break
            pdf.showPage()
        pdf.save()
    marker.write_text(json.dumps(spec))
    print(f"Generated {args.pages} pages in {file_no} PDFs ({time.perf_counter() - started:.0f}s)")


def write_markdown(corpus_dir: Path, paragraphs: list[str], args) -> None:
    """--no-pdf: the same volume of text as markdown, one file per 'page'."""
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True)
    rng = random.Random(args.seed)
    for i, page in enumerate(page_texts(paragraphs, args.pages, args.chars_per_page, rng), start=1):
        (corpus_dir / f"page-{i:05d}.md").write_text(f"# Page {i}\n\n" + "\n\n".join(page), encoding="utf-8")


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        # Untracked files (like earlier result JSONs) do not change the code under test.
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scale benchmark for the ingest + retrieval pipeline.")
    parser.add_argument("--pages", type=int, default=12_000)
    parser.add_argument("--pages-per-file", type=int, default=500)
    parser.add_argument("--chars-per-page", type=int, default=3_000, help="~a dense technical-manual page")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-pdf", action="store_true", help="markdown instead of PDFs (skips extraction)")
    parser.add_argument("--embedding", choices=["fake", "openai"], default="fake")
    parser.add_argument("--rerank", action="store_true", help="include the cross-encoder stage in query latency")
    parser.add_argument("--workers", type=int, default=None, help="loader processes (default: auto)")
    parser.add_argument("--source-dir", default=config.DATA_DIR, help="markdown to sample page text from")
    parser.add_argument("--corpus-dir", default=str(HERE / ".corpus"))
    parser.add_argument("--work-dir", default=str(HERE / ".work"))
    parser.add_argument("--label", default="", help="tag for the results file, e.g. before-m2")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    # Imported late so --help works without the heavy stack.
    import app.store as store_mod
    from app.chunking import structured_split
    from app.loaders import load_directory
    from app.retriever import build_retriever

    # Throughput is the pipeline's, not a provider's rate limit.
    config.EMBED_RPM = 0
    config.EMBED_BATCH_SIZE = 256 if args.embedding == "fake" else 100
    config.RERANK = args.rerank
    if args.embedding == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        fake = DeterministicFakeEmbedding(size=1536)
        store_mod.get_embeddings = lambda *a, **k: fake
        config.EMBED_PROVIDER, config.EMBEDDING_MODEL = "fake", "fake-1536"
    else:
        config.EMBED_PROVIDER, config.EMBEDDING_MODEL = "openai", "text-embedding-3-small"
        config.require_embed_key("openai")

    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    config.EMBED_CACHE_DIR = str(work / "embed_cache")  # cold cache: embed everything
    index_dir = str(work / "index")

    paragraphs = source_paragraphs(Path(args.source_dir))
    corpus_dir = Path(args.corpus_dir) / ("md" if args.no_pdf else "pdf")
    if args.no_pdf:
        write_markdown(corpus_dir, paragraphs, args)
    else:
        write_pdfs(corpus_dir, paragraphs, args)

    stages: dict[str, dict] = {}
    load_kwargs = {"workers": args.workers} if "workers" in inspect.signature(load_directory).parameters else {}

    with MemoryMonitor() as mem:

        def stage(name: str, fn):
            gc.collect()
            mem.stage_peak = mem.rss()
            started = time.perf_counter()
            result = fn()
            seconds = time.perf_counter() - started
            stages[name] = {"seconds": round(seconds, 2), "peak_rss_mb": _mb(mem.stage_peak), "rss_after_mb": _mb(mem.rss())}
            print(f"  {name:<14} {seconds:8.1f}s   peak {_mb(mem.stage_peak):8.0f} MB")
            return result

        print(f"\nPipeline ({args.embedding} embeddings, {config.summary()})")
        baseline_rss = mem.rss()
        docs = stage("load", lambda: load_directory(corpus_dir, **load_kwargs))
        pdf_pages = sum(1 for d in docs if d.metadata.get("page"))
        chunks = stage("chunk", lambda: structured_split(docs))
        store = stage("embed+index", lambda: store_mod.build_index(chunks, show_progress=False))
        stage("save", lambda: store_mod.save_index(store, chunks, index_dir))
        ntotal = store.index.ntotal
        del store, docs
        store = stage("load_index", lambda: store_mod.load_index(index_dir))
        chunks = store_mod.load_chunks(index_dir)
        retriever = stage("build_retriever", lambda: build_retriever(store, chunks))

        rng = random.Random(args.seed + 1)
        queries = [" ".join(rng.choice(paragraphs).split()[:12]) for _ in range(args.queries)]
        for q in queries[:5]:  # warm-up: first-call allocations are not steady state
            retriever.invoke(q)
        latencies = []

        def run_queries():
            for q in queries:
                t = time.perf_counter()
                retriever.invoke(q)
                latencies.append((time.perf_counter() - t) * 1000)

        stage("queries", run_queries)
        steady_rss = mem.rss()

    latencies.sort()
    index_bytes = sum(p.stat().st_size for p in Path(index_dir).rglob("*") if p.is_file())
    ingest_seconds = sum(stages[s]["seconds"] for s in ("load", "chunk", "embed+index", "save"))
    result = {
        "label": args.label,
        "git": git_state(),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        },
        "params": {k: v for k, v in vars(args).items() if k not in {"source_dir", "corpus_dir", "work_dir"}},
        "corpus": {"format": "md" if args.no_pdf else "pdf", "pdf_pages": pdf_pages, "files": len(list(corpus_dir.glob("*.*"))) - (0 if args.no_pdf else 1)},
        "chunks": len(chunks),
        "faiss_vectors": ntotal,
        "index_on_disk_mb": _mb(index_bytes),
        "ingest_seconds": round(ingest_seconds, 1),
        "stages": stages,
        "peak_rss_mb": _mb(mem.peak),
        "baseline_rss_mb": _mb(baseline_rss),
        "serving_rss_mb": _mb(steady_rss),
        "query_latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p95": round(latencies[int(len(latencies) * 0.95) - 1], 1),
            "max": round(latencies[-1], 1),
            "mean": round(statistics.fmean(latencies), 1),
            "n": len(latencies),
        },
    }

    print(
        f"\n{result['corpus']['pdf_pages'] or args.pages} pages -> {result['chunks']} chunks | "
        f"ingest {result['ingest_seconds']}s | peak RSS {result['peak_rss_mb']:.0f} MB | "
        f"serving RSS {result['serving_rss_mb']:.0f} MB | "
        f"query p50 {result['query_latency_ms']['p50']} ms, p95 {result['query_latency_ms']['p95']} ms"
    )

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"scale-{args.label + '-' if args.label else ''}{time.strftime('%Y%m%d-%H%M%S')}.json"
        (RESULTS_DIR / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved bench/results/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
