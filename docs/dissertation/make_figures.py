"""Draw the dissertation's figures from the logged measurement files.

    ragenv311\\Scripts\\python docs/dissertation/make_figures.py

Inputs (docs/dissertation/data/): latency_*.json written by
eval/latency_probe.py, and benchmarks.json from the component benchmarks.
Every plotted number comes from those files; nothing is typed in here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"

INK, ACCENT, SOFT, WARM = "#1F2A44", "#2F6F8F", "#DCE9F0", "#C8553D"


def _runs() -> list[dict]:
    runs = []
    for path in sorted(DATA.glob("latency_*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        run["_file"] = path.name
        runs.append(run)
    return runs


def _latest(runs: list[dict], backend: str) -> dict | None:
    matching = [r for r in runs if r.get("backend") == backend and r.get("questions")]
    return matching[-1] if matching else None


def architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, fill=SOFT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fill, ec=INK, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     color=INK, lw=1.0))

    box(0.2, 2.2, 1.6, 0.9, "Browser\nNext.js UI", "#F3E9DC")
    box(2.3, 2.2, 1.7, 0.9, "FastAPI\nNDJSON stream\n+ heartbeat")
    # indexing lane
    ax.text(4.6, 4.85, "Indexing", fontsize=9.5, weight="bold", color=ACCENT)
    box(4.5, 3.6, 1.6, 0.9, "PDFExtractor\nnative · tables\nOCR fallback")
    box(6.5, 3.6, 1.7, 0.9, "Structure-aware\nchunking")
    box(8.6, 4.15, 2.1, 0.55, "FTS5 lexical index")
    box(8.6, 3.45, 2.1, 0.55, "Vector index (e5)")
    # question lane
    ax.text(4.6, 1.75, "Answering", fontsize=9.5, weight="bold", color=ACCENT)
    box(4.5, 0.5, 1.5, 0.9, "Intent\nrouter")
    box(6.3, 0.5, 1.4, 0.9, "Cross-encoder\nrerank")
    box(8.0, 0.5, 1.2, 0.9, "Gate +\nprompt")
    box(9.5, 0.5, 1.3, 0.9, "LLM\nOllama / Groq", "#F3E9DC")

    arrow(1.8, 2.65, 2.3, 2.65)
    arrow(4.0, 2.85, 4.5, 3.9)
    arrow(6.1, 4.05, 6.5, 4.05)
    arrow(8.2, 4.2, 8.6, 4.4)
    arrow(8.2, 3.9, 8.6, 3.7)
    arrow(4.0, 2.45, 4.5, 1.1)
    arrow(6.0, 0.95, 6.3, 0.95)
    arrow(7.7, 0.95, 8.0, 0.95)
    arrow(9.2, 0.95, 9.5, 0.95)
    arrow(9.6, 3.45, 5.4, 1.4)  # router reads the indexes
    ax.text(7.6, 2.55, "lookups / RRF", fontsize=8, color=ACCENT, rotation=-27)
    return _save(fig, "fig_architecture.png")


def answer_latency(run: dict) -> Path:
    rows = [q for q in run["questions"] if q.get("result") == "done"]
    labels = [f"{q['folder']}: {q['question'][:46]}" for q in rows]
    retrieval = [(q.get("retrieval_ms") or 0) / 1000 for q in rows]
    rerank = [(q.get("rerank_ms") or 0) / 1000 for q in rows]
    generation = [(q.get("generation_ms") or 0) / 1000 for q in rows]
    first_token = [q.get("first_token_s") for q in rows]

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.6))
    y = range(len(rows))
    ax.barh(y, retrieval, color=ACCENT, label="retrieval + rerank")
    if any(rerank):  # only when the pipeline, not the router, reranked
        ax.barh(y, rerank, left=retrieval, color="#7FA7BA", label="rerank (pipeline)")
    left = [a + b for a, b in zip(retrieval, rerank)]
    ax.barh(y, generation, left=left, color="#E3B23C", label="generation")
    ax.scatter([t for t in first_token if t], [i for i, t in zip(y, first_token) if t],
               marker="|", s=160, color=WARM, zorder=3, label="first token (client)")
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    engine = rows[0].get("backend"), rows[0].get("model")
    ax.set_title(f"Answer latency per question — {engine[0]} · {engine[1]} (run {run['_file']})",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "fig_answer_latency.png")


def rerankers(bench: dict) -> Path:
    data = bench["reranker_ms_per_15_candidates"]
    models = list(data)
    lengths = ["max_length_384", "max_length_256"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.36
    for i, length in enumerate(lengths):
        values = [data[m][length] / 1000 for m in models]
        bars = ax.bar([x + (i - 0.5) * width for x in range(len(models))], values, width,
                      color=[ACCENT, "#7FA7BA"][i], label=length.replace("max_length_", "max length "))
        ax.bar_label(bars, fmt="%.1f s", fontsize=8)
    ax.set_xticks(range(len(models)), [m.split("/")[-1] for m in models], fontsize=8.5)
    ax.set_ylabel("seconds per question (15 candidates)")
    ax.set_title("Cross-encoder reranking cost on CPU", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig_rerankers.png")


def index_time(runs: list[dict]) -> Path | None:
    """Extraction vs embedding time per folder, from the first (cold) index run."""
    folders: dict[str, dict] = {}
    for run in runs:
        for entry in run.get("index_runs", []):
            folder = entry["index"]["folder"]
            if folder in folders or entry["index"].get("result") != "done":
                continue
            extraction = embedding = 0.0
            for line in entry.get("log", []):
                if m := re.search(r"page\(s\) in ([\d.]+)s", line or ""):
                    extraction += float(m.group(1))
                if m := re.search(r"Embedded \d+ chunks in ([\d.]+)s", line or ""):
                    embedding += float(m.group(1))
            if embedding:
                folders[folder] = {"extraction": extraction, "embedding": embedding,
                                   "total": entry["index"]["total_s"]}
    if not folders:
        return None
    names = list(folders)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ext = [folders[n]["extraction"] for n in names]
    emb = [folders[n]["embedding"] for n in names]
    other = [max(folders[n]["total"] - e - m, 0) for n, e, m in zip(names, ext, emb)]
    ax.barh(names, ext, color="#7FA7BA", label="extraction")
    ax.barh(names, emb, left=ext, color=ACCENT, label="embedding")
    ax.barh(names, other, left=[a + b for a, b in zip(ext, emb)], color="#D9D9D9", label="other")
    for i, n in enumerate(names):
        ax.text(folders[n]["total"] + 3, i, f"{folders[n]['total']:.0f} s", va="center", fontsize=8.5)
    ax.set_xlabel("seconds (first index)")
    ax.set_title("Where first-index time goes", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "fig_index_time.png")


def run_comparison(runs: list[dict]) -> Path | None:
    """Median first-token time and prompt size for each labelled local-model run."""
    import statistics

    labels = json.loads((DATA / "run_labels.json").read_text(encoding="utf-8"))
    rows = []
    for run in runs:
        answered = [q for q in run.get("questions", []) if q.get("result") == "done"]
        if run.get("backend") != "ollama" or not answered or run["_file"] not in labels:
            continue
        rows.append((
            labels[run["_file"]],
            statistics.median(q["first_token_s"] for q in answered if q.get("first_token_s")),
            statistics.median(q["input_tokens"] for q in answered),
            sum(q.get("decision") == "answer" for q in answered),
            len(answered),
        ))
    if not rows:
        return None
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 0.6 * len(rows) + 1.8), sharey=True)
    names = [r[0] for r in rows]
    bars = left.barh(names, [r[1] for r in rows], color=ACCENT)
    left.bar_label(bars, fmt="%.1f s", fontsize=8)
    left.set_xlabel("median time to first token (s)")
    bars = right.barh(names, [r[2] for r in rows], color="#7FA7BA")
    right.bar_label(bars, labels=[f"{r[2]:.0f} tok · {r[3]}/{r[4]} answered" for r in rows], fontsize=8)
    right.set_xlabel("median prompt tokens")
    left.invert_yaxis()
    for ax in (left, right):
        ax.grid(axis="x", alpha=0.3)
        ax.margins(x=0.35)  # room for the value labels beside each bar
    fig.suptitle("Local model (qwen2.5:3b): prompt configuration vs latency and answers", fontsize=10)
    return _save(fig, "fig_run_comparison.png")


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"wrote {path.relative_to(HERE)}")
    return path


def main() -> int:
    runs = _runs()
    bench = json.loads((DATA / "benchmarks.json").read_text(encoding="utf-8"))
    architecture()
    rerankers(bench)
    index_time(runs)
    run_comparison(runs)
    for backend in ("ollama", "groq"):
        run = _latest(runs, backend)
        if run:
            answer_latency(run)
            break
    else:
        print("no latency run with a real engine yet; skipped fig_answer_latency.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
