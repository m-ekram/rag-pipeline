# STATE.md — where this project is

Updated 2026-09-02, after finishing the six milestones planned from the original
orientation pass. Everything below was executed, not inferred.

**The pipeline is complete and verified end to end**, on the host and in Docker.
The original audit found a working pipeline that could not survive contact with
a different machine; that is what got fixed.

---

## Pipeline, stage by stage

| Stage | Where | Status |
|---|---|---|
| load | `app/loaders.py` | pdf/md/rst/txt/html/docx → Documents with `source`, `title`, `page`. Per-file failure isolation. |
| chunk | `app/chunking.py` | Structure-aware splits + contextual headers (ships); fixed-width naive splitter (the eval baseline). |
| embed | `app/store.py` | Batched, retrying, rate-limited, disk-cached by text hash so an interrupted run resumes. |
| store | `app/store.py` | FAISS + `chunks.jsonl` sidecar for BM25 + `index_meta.json`. Refuses to open an index built with a different embedding model. |
| retrieve | `app/retriever.py` | Hybrid BM25 + dense. All knobs take explicit overrides, defaulting to config. |
| rerank | MMR | Present, measured, **off by default** (`MMR_LAMBDA=1.0`). No cross-encoder exists. |
| generate | `app/rag.py` | condense → retrieve → ground → answer, sync and streaming. |
| cite | `app/attribution.py` | Computed attribution when the model emits no markers. |
| serve | `app/api.py` | FastAPI, 10 routes, request *and* response schemas. |
| measure | `eval/evaluate.py` | hit@k / MRR / precision@k, per failure class, with a committed baseline. |

## Verified numbers

Corpus pinned at FastAPI `0.141.1` @ `95f8322ee1dc`, 141 files, 1,534 chunks.

```
config                chunks     hit@5       MRR   precision
baseline                1081    75.0%     0.635      42.8%
tuned                   1534    83.3%     0.699      46.7%
```

| class | baseline | tuned |
|---|---|---|
| exact | 12/12 | 12/12 |
| paraphrase | 8/12 | 10/12 |
| multihop | 7/12 | 8/12 |

Ablation (one shared index, retriever varied alone):

```
dense-only  83.3%   dense+mmr  77.8%   hybrid  83.3%
hybrid+mmr  77.8%   l=0.8      77.8%   sparse-heavy  36.1%
```

- Host: index loads in 17 s, retrieval 0.2 s, answer ~50-75 s on local Qwen2.5-3B.
- Docker: image 2.48 GB, ingest 286 s, `/ask` 66.5 s.
- `pytest`: 215 tests, ~10 s, fully offline.
- `python -m eval.evaluate --check`: deltas of exactly `+0.000`.

## What was broken, and what fixed it

| Problem | Fix |
|---|---|
| Clean checkout died on the first command — `.env.example` said `google`, README promised keyless local | `.env.example` is genuinely local-first; verified by running a clean checkout with no key |
| No tests at all; pytest not installed | 215 offline tests, `requirements-dev.txt` |
| Corpus cloned an unpinned branch; the checkout that produced it was gone | Pinned `--ref` (default `0.141.1`) + `CORPUS.lock.json` + `--verify` |
| Ablation variants inherited ambient `MMR_LAMBDA`, so every `+mmr` row silently ran with MMR off | Variants declare their own settings; retrievers take explicit params instead of the harness mutating globals |
| No baseline checked in — numbers lived only in README prose | `eval/baseline.json` with corpus provenance, plus `--check` |
| Endpoints annotated `-> dict`; no response shape in OpenAPI | Six response models wired via `response_model` |
| Docker image could not build (CUDA torch, no llama-cpp wheel, no compiler) | CPU torch index + prebuilt llama-cpp wheel index; verified by building and running |
| `--rebuild` did `rmtree` on the Docker volume mount point → EBUSY | Clears contents, keeps the directory |
| Embed-cache volume was root-owned, unwritable by the container user | Directory created in the image so the volume inherits ownership |
| README quick start unfollowable; `/ask` example cited a nonexistent path | Rewritten and verified from a clean checkout, with real response values |

## Definition of done

| Requirement | Status |
|---|---|
| Ingestion reproducible from a clean checkout on a documented corpus | **Done** — pinned ref, lock file, `--verify`, byte-identical across rebuilds |
| Retrieval measurable, one command, baseline checked in | **Done** — `python -m eval.evaluate --check`, `eval/baseline.json` |
| …with answer-quality scores | **Cut by decision (2026-09-02).** Retrieval is the only measured axis, stated plainly in the README |
| FastAPI service with query + health and request/response schemas | **Done** |
| Dockerfile + compose bringing up the app and its store | **Done** — FAISS in-process by decision; compose runs the app plus index/model/cache volumes |
| README a stranger can follow, plus the eval numbers | **Done** — verified by following it in a clean checkout |
| Config in env vars, no secrets or absolute paths | **Done** — greps for absolute paths and key patterns are clean; `.env` untracked |

## Deliberate omissions

- **Answer-quality scoring.** Cut, not forgotten. The README says so explicitly
  rather than letting the eval look more complete than it is.
- **MMR.** Measured as harmful here and disabled. The code stays as the evidence
  for a documented finding; no further investment.
- **A separate vector store service.** FAISS in-process is correct at 1,534
  chunks; a network hop would add operational surface and buy nothing.
- **Incremental indexing.** Ingest still rebuilds wholesale. Fine for a corpus
  that changes on a release cadence; wrong for one that changes hourly.

## Known limits

- Local answer latency (~50-75 s) is a property of this CPU and
  `LLAMA_N_THREADS`, not a portable number.
- One FastAPI include directive points at a file absent at `0.141.1`; the corpus
  builder reports it (439 of 440 inlined).
- `langchain-community` emits a sunset deprecation warning on import.
- Six golden questions still miss, all `paraphrase` or `multihop`. Exact-identifier
  retrieval is solved; that is where the remaining headroom is.
