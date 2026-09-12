# Session handoff — 2026-09-12

## Current task (user's latest request)

> "Add the numbers needed to make this dissertation acceptable at a university,
> run this project to generate them, then update the dissertation."

Plan agreed in-session — three evaluations, all numbers produced by scripts:

| # | Evaluation | Script | Output |
|---|---|---|---|
| E1 | FiQA retrieval benchmark: BM25 / dense / hybrid RRF / hybrid+rerank; P@5, R@10, R@100, MRR@10, nDCG@10, Hit@5; bootstrap 95% CIs; paired randomization tests; latency per query; contamination ρ ∈ {0, 0.5, 0.9} (nested) | `eval/benchmark_fiqa.py` → `eval/analyse_fiqa.py` | `docs/dissertation/data/fiqa_benchmark.json`, `…_summary.json` |
| E2 | Selective retrieval / abstention: risk–coverage + AURC per confidence signal (reranker, RRF, dense, BM25); threshold calibrated on **dev** (500 q) at target coverage 0.8/0.6, reported on **test** (648 q); drift = ρ0 threshold applied at ρ0.5/0.9 | same two scripts | same summary file |
| E3 | End-to-end answer quality on own documents: 30 labelled questions (14 Master Plan, 10 paper, 6 unanswerable), accuracy, citation-page precision, abstention on answerable/unanswerable, latency mean/sd/median/p90, Wilson 95% CIs | `eval/answer_eval.py` + `eval/labelled_questions.json` | `docs/dissertation/data/answer_eval_<stamp>.json` |

## State

**Branch** `refactor/optimization`, last pushed commit `df655a3` (docs). All
earlier work (9 commits) is on GitHub. The evaluation tooling below is
**committed locally but not pushed**, and **not yet run in full**:

- `eval/benchmark_fiqa.py` — E1/E2 runner (nested levels, embeds once, caches rerank scores).
- `eval/analyse_fiqa.py` — statistics, calibration, drift; writes `*_summary.json`, prints Markdown tables.
- `eval/answer_eval.py`, `eval/labelled_questions.json` — E3. Expected answers/pages were read from the extracted document text (see the JSON's `_comment`).
- `retrieval/local_dense.py` — added read-only `chunk_ids` / `matrix` properties (+ test in `tests/test_local_dense.py`, passing).
- `docs/dissertation/make_figures.py` — new figures `fig_fiqa_retrieval`, `fig_fiqa_risk_coverage`, `fig_fiqa_signals`, `fig_answer_quality`, drawn only when their data files exist.

**Running when this was written** (background tasks do not survive the session):

- Smoke run of E1: `--rhos 0 --max-queries 15 --out <scratchpad>\fiqa_smoke.json`. It had embedded the level-0 corpus (2,944 judged docs → 4,293 chunks, 455 s ≈ 9.4 chunks/s) into `.cache/vectors/fiqa_eval_multilingual-e5-small/`; the full run reuses that prefix.
- FastAPI on :8000 (production UI served from `web/out`), Ollama on :11434 with `qwen2.5:3b` resident.

## Next steps (in order)

Use the repaired venv: `ragenv311\Scripts\python.exe` (from `C:\dev\rag-pipeline`).

1. ~~Smoke-check~~ **Done.** Smoke run (ρ = 0, 15 test + 15 dev queries) and
   `analyse_fiqa.py` both completed; every table, test, AURC, calibration and
   drift line was produced. Smoke nDCG@10: BM25 0.353, dense 0.628, hybrid
   0.565, hybrid+rerank 0.528 (n = 15, CIs ±0.2 — not reportable). **Verify in
   the full run:** if dense ≥ hybrid holds at 648 queries, that is a genuine
   finding to report (weak BM25 on FiQA dilutes RRF; the English ms-marco
   reranker may not add to multilingual e5) — report it as found, don't tune it
   away.
2. **Full E1/E2** (~1.5–2 h, CPU-bound; run in background, nothing else heavy alongside):
   `python -u eval/benchmark_fiqa.py --out docs/dissertation/data/fiqa_benchmark.json`
   Levels: ρ = 0, 0.5, 0.9 → 2,944 / 5,888 / 29,440 docs. Saves after each level.
3. **Analyse:** `python eval/analyse_fiqa.py docs/dissertation/data/fiqa_benchmark.json`
4. **E3** (after 2 finishes, so latencies are not contended; server must be running):
   `python -u eval/answer_eval.py --backend ollama --out docs/dissertation/data` (~20–25 min).
   If accept strings mis-score a clearly correct answer, fix the *question file* and say so in the dissertation — do not edit results.
5. `python docs/dissertation/make_figures.py`
6. **Update the dissertation** (Markdown in `docs/dissertation/chapters/`):
   - Ch.4 Methodology: add FiQA (BEIR), splits (dev = calibration, test = reporting), metrics, bootstrap CIs, randomization tests, contamination levels, selective-risk definition (risk = 1 − Hit@5 of hybrid+rerank), the labelled-question construction (author-written, single annotator), Wilson intervals.
   - Ch.6 Results: new sections for E1 (table + `fig_fiqa_retrieval`, significance), E2 (`fig_fiqa_risk_coverage`, `fig_fiqa_signals`, calibration and drift table), E3 (`fig_answer_quality`, accuracy/citation/abstention/latency table).
   - Ch.7 Limitations: labelled set is small and author-written; FiQA is English-only; one embedding model; rolls not measured (OCR).
   - Abstract, Ch.8 Conclusion, `docs/ARCHITECTURE.md` §7: headline numbers.
   - `python docs/dissertation/build_docx.py`; check no placeholders (inspect with python-docx as done before).
7. `python -m pytest -q -p no:cacheprovider` (all must pass; 294 at last run), then commit and ask before pushing.

## Environment facts and decisions (don't re-litigate)

- **Venv**: `ragenv311/pyvenv.cfg` repointed to `C:\Program Files\Python311` (backup `pyvenv.cfg.bak`). Through the venv launcher, `qdrant_client` imports; no Qdrant server runs, so `LocalDenseIndex` is used.
- **Engines**: user skipped Groq; Ollama installed with `qwen2.5:3b` (default) and `qwen2.5:1.5b` (rejected: answered 40% for 55.04%).
- **OCR**: PaddlePaddle 3.3.1 broken on this Windows (oneDNN error; ~110 s/page without it); 3.0.0 conflicts with torch. Tesseract install needs UAC — user chose **"skip rolls on this laptop"**. Tesseract language data is in `.cache/tessdata`, `pytesseract` installed.
- **Final local prompt config** = run D: compact system prompt + "answer in sentences" rule, evidence budget 1,200 tokens with parent sections.
- **User preferences**: no Claude attribution in commits/PRs; push only after asking.

## Numbers already in the dissertation (for consistency)

- Master Plan first index 317 s (316 s embedding, 1,616 chunks); re-index ~7 s.
- Retrieval+rerank 2.9–3.7 s → 0.8–1.2 s; reranker L6 ~1.0 s, L12 4.3→2.2 s per 15 candidates.
- qwen2.5:3b ~29 prompt tok/s, ~6.5 output tok/s; final median first token 31.6 s, total 38.5 s (run D, 10 golden questions).
- Runs A–D table in Ch.6 and `ARCHITECTURE.md` §7.1.

## Gotchas

- The Edit tool fails to match long `old_string`s containing Urdu/Devanagari or `\uXXXX` text — anchor edits on ASCII-only lines.
- PowerShell `Remove-Item Env:X` is blocked by the sandbox — use `$env:X = $null`.
- Python output to a pipe is buffered — run long scripts with `python -u` to watch progress.
- `ingestion/pdf_extractor.py` is stored with CRLF; keep it CRLF or the diff becomes whole-file.
- Restart the server after code changes: stop the process on :8000, then `Start-Process ragenv311\Scripts\python.exe -ArgumentList '-m','uvicorn','api.server:app','--port','8000'`.
