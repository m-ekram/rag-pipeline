# STATE.md — where this project actually is

Written after a read-only orientation pass on 2026-09-02, against commit
`e8e30a3`. Everything marked *verified* below was executed; everything marked
*inferred* was read but not run, and is labelled as such.

**Headline: this is a working pipeline, not a scaffold.** Every stage from
ingest to serve runs end to end on this machine. There are no stubs, no
`TODO`/`FIXME` markers, no absolute paths, and no mock data dressed up as a
feature. What is missing is everything that makes it survive contact with a
different machine: tests, a reproducible corpus, regenerable numbers, and a
Docker image that builds.

---

## 1. End-to-end trace

### ingest — `app/ingest.py` · complete
`python -m app.ingest [--rebuild] [--dry-run] [--chunk-size N] [--chunk-overlap N]`

Loads `DATA_DIR`, chunks, embeds, writes the index. `--dry-run` skips embedding
and therefore skips the API-key check. Prints chunk statistics and elapsed time.

**Verified:** `python -m app.ingest --dry-run` → 141 files → 141 sections →
1533 chunks, chars min=83 median=770 mean=692 max=1000.

### load — `app/loaders.py` · complete
`.pdf .md .markdown .txt .rst .html .htm .docx`. PDFs split per page via `pypdf`
with per-page failure isolation; HTML strips `script/style/nav/footer` via
BeautifulSoup; docx via python-docx; everything else read as plain text.
`clean_text` rejoins hyphen-broken words, collapses whitespace runs, and caps
blank lines at two. Every Document carries `source` (path relative to the data
root), `title`, and `page` for PDFs.

Errors are handled the right way: a single bad file logs a warning and is
skipped; an empty or missing data directory is a hard `SystemExit` with an
actionable message.

### chunk — `app/chunking.py` · complete
Two splitters, both real and both used:
- `structured_split` (ships) — `RecursiveCharacterTextSplitter` over
  markdown-aware separators (`\n## `, `\n### `, `\n\n`, `\n`, `. `, ` `, `""`),
  with a contextual header `[Title - p.12]` prepended to each chunk before
  embedding.
- `naive_split` (baseline) — fixed-width windows, no overlap, no headers. Not
  dead code; it is what `eval/evaluate.py` measures against.

`_finalise` drops fragments under `MIN_CHUNK_CHARS` (80) and stamps
`chunk_index`, `chunk_id` (sha1 of source+index+first 200 chars), `char_count`.

### embed + store — `app/store.py` · complete, and the most carefully built file here
- Batched embedding with exponential backoff, jitter, and preference for the
  provider's own `retryDelay` hint over its own guess.
- `_RateLimiter` tracks individual request timestamps in a rolling 60s window
  rather than spacing batches by average cost. The docstring explains why both
  naive approaches fail. This is real engineering against a real quota wall.
- Disk-backed vector cache keyed by sha1 of chunk text, flushed per batch, so an
  interrupted run resumes instead of re-paying. Tolerates a torn final line.
- `save_index` writes the FAISS index, a `chunks.jsonl` sidecar (needed because
  BM25 requires the raw documents), and `index_meta.json`.
- `load_index` refuses to open an index whose recorded embedding model differs
  from current config, rather than silently returning garbage. Good instinct.

**Verified:** index loads in 17.1s, `chunk_count` 1533, built 2026-08-25.

### retrieve — `app/retriever.py` · complete
Dense leg is FAISS `search_type="mmr"` with `k`, `fetch_k=max(FETCH_K, k*4)`,
`lambda_mult=MMR_LAMBDA`. When `USE_HYBRID`, an `EnsembleRetriever` fuses it with
BM25 at weights (0.6, 0.4). Import path for `EnsembleRetriever` falls back across
`langchain_classic` → `langchain`; missing BM25 degrades to dense-only with a
warning rather than crashing.

**Verified:** "How do I use BackgroundTasks?" returned in 0.2s with
`tutorial__background-tasks.md` in slots 1, 3, 4 and `reference__background.md`
in slot 2. Correct.

Note: the ensemble returns up to `2k` documents (10 at k=5); both callers slice
to `k` afterwards. Correct as written, but the retriever's own `k` is not the
number of documents it hands back.

### rerank — MMR · present, measured, deliberately disabled
`MMR_LAMBDA` defaults to **1.0**, which makes MMR mathematically equivalent to
plain similarity. This was an evidence-driven decision, documented in
`config.py:83-88`: at λ=0.5 it cost 5.5 points of hit@5 and 20 points of
precision, because sibling chunks of one long page are usually *all* relevant.
The code is kept as the evidence for the finding. **There is no cross-encoder
reranker** — "rerank" in this pipeline means MMR and nothing else.

### generate — `app/rag.py` · complete
`condense → retrieve → ground → answer`. History-bearing questions are rewritten
into standalone search queries first, with a try/except so a failed rewrite falls
back to the raw question instead of breaking the answer. The answer prompt is
strict about grounding and citation format. `NO_CONTEXT_MESSAGE` is returned when
retrieval comes back empty. Both sync `ask` and async `astream` exist; the
streaming variant emits `sources`/`token`/`replace`/`done`/`error` events.

**Verified:** end-to-end answer in 74.6s on local Qwen2.5-3B. Grounded,
on-corpus, correctly cited, with real source paths.

### citations — `app/attribution.py` · complete, and honest
`CITATION_MODE=auto` (the local default) computes attribution when the model
emits no `[n]` markers: each sentence is embedded and matched against the
passages actually retrieved; the best match above `CITE_THRESHOLD` (0.55) becomes
its marker; no confident match means no marker. Fenced code is never annotated;
headings, table rows, and bare bullets are skipped; the marker is placed before
trailing punctuation.

**I specifically tried to catch this faking it.** Re-ran the same question with
`CITATION_MODE=off`: the model emitted **zero** `[n]` markers. So the computed
path is genuinely what supplies them, the README's claim about small models is
accurate, and the docstring's framing — that a computed citation is a weaker
claim than a model-asserted one — is the correct way to describe it.

### serve — `app/api.py` + `app/ui/index.html` · complete
Ten routes. Engine loads once in `lifespan`; a failed load is captured and served
as a 503 with the real reason rather than crashing the process. `/ask` is
dispatched via `asyncio.to_thread` so the sync chain does not block the loop.
`/reload` re-opens the index after a re-ingest. 281-line dependency-free
streaming chat UI.

**Verified:** module imports; routes `/openapi.json /docs /docs/oauth2-redirect
/redoc /health /sources /ask /ask/stream /reload /`.

### measure — `eval/evaluate.py` · complete but with a reproducibility bug (§3.3)
36 golden questions tagged `exact` / `paraphrase` / `multihop`. Scoring requires
both the right source file *and* (when given) a `must_contain` string, so a large
file cannot score a hit just for being the right file. Modes: `--compare`,
`--config`, `--sweep`, `--ablate`, `--validate`, `--save`.

**Verified:** `--validate` → all 36 questions reachable in the corpus.

---

## 2. Stage status summary

| Stage | Status |
|---|---|
| load, chunk, embed, store, retrieve, generate, cite, serve, eval | **Complete and verified running** |
| rerank (MMR) | Present, measured, deliberately off. No cross-encoder exists. |
| **tests** | **Do not exist at all** — no `tests/`, no pytest, no runner |
| **answer-quality scoring** | **Does not exist** — cut from scope 2026-09-02 |
| **reproducible corpus** | **Does not exist** — corpus is unpinned (§3.2) |
| **checked-in baseline** | **Does not exist** — `eval/results/` is gitignored |
| **API response schemas** | **Do not exist** — every endpoint returns bare `dict` |
| Docker | Files exist; the image is **inferred not to build** (§3.5) |

---

## 3. What is broken

### 3.1 A clean checkout fails on the first command — *verified*
`.env.example:4` sets `LLM_PROVIDER=google` and leaves `GOOGLE_API_KEY` empty,
while `README.md:68-69` says "The defaults are fully local, so no key is needed."
Both cannot be true. Following the README literally:

```
$ cp .env.example .env && python -m app.ingest
SystemExit: GOOGLE_API_KEY is not set (EMBED_PROVIDER=google).
```

`config.py:30` also defaults `PROVIDER` to `"google"`, so the code agrees with
`.env.example` and disagrees with the README.

### 3.2 The corpus cannot be reproduced — *verified*
`scripts/prepare_fastapi_docs.py` clones FastAPI `main` at `--depth 1` with no
pinned ref. Worse, the `.cache/fastapi` checkout is **gone from disk**, so the
provenance of the current 141-file `data/fastapi/` is unrecoverable. Re-running
the script today clones whatever `main` is now and may produce a different
corpus — and therefore different numbers — with no way to tell that it drifted.
`data/` is gitignored (correctly), so the corpus ships as a recipe, and the
recipe is not deterministic.

### 3.3 The published ablation table cannot be regenerated — *verified by reading*
`eval/evaluate.py:90`:
```python
config.MMR_LAMBDA = spec.get("lambda", 1.0 if not spec["mmr"] else original[1])
```
`original[1]` is ambient `config.MMR_LAMBDA`, which now defaults to **1.0** (MMR
off). So `dense+mmr` and `hybrid+mmr` declare `mmr: True` but silently run with
MMR disabled, making them identical to `dense-only` and `hybrid`. The README's
table (dense+mmr 77.8%, hybrid+mmr 77.8%) was measured when the default was 0.5.
Re-running `--ablate` from a clean checkout today would print four identical rows
and quietly erase the very finding the table exists to document.

The root cause is that eval varies configuration by **mutating module-level
globals in `config` and restoring them in a `finally`**, so a variant can inherit
an unrelated ambient default instead of declaring its own.

### 3.4 No baseline number is checked in — *verified*
`.gitignore` excludes `eval/results/`. The three result files on disk are
untracked. The only record of the project's numbers is README prose. Nothing can
detect a regression.

### 3.5 Docker almost certainly does not build — *inferred, not verified*
Reading only; I did not run `docker build`. Four independent problems:
1. `llama-cpp-python` has no prebuilt manylinux wheel on PyPI and
   `python:3.11-slim` ships no compiler, so `pip install -r requirements.txt`
   should fail outright. The correct wheel index is written as a comment in
   `requirements.txt:15` but never applied to the Dockerfile.
2. **`torch` is not in `requirements.txt` at all.** It arrives transitively via
   `sentence-transformers`, and the PyPI default on Linux is the CUDA build —
   GBs of unnecessary image. The CPU index URL is, again, only a comment
   (`requirements.txt:9-10`). This venv has `torch 2.13.0+cpu`, which
   `requirements.txt` alone would not have produced.
3. The image never copies `models/`, and compose mounts no model volume, so the
   documented local default cannot run in a container.
4. No `HF_HOME` volume, so `bge-small-en-v1.5` re-downloads on every fresh
   container.

Also a likely runtime trap: the container runs as uid 10001 (`Dockerfile:22`)
but `docker-compose.yml:11` bind-mounts host-owned `./faiss_index` read-write.
On a Linux host, `docker compose run --rm ingest` should fail to write the index.

### 3.6 No API response schemas — *verified*
`/health`, `/sources`, `/ask`, `/reload` are annotated `-> dict`. Requests are
properly modelled (`AskRequest`, `Turn`, with `min_length`/`max_length`/`ge`/`le`
and a role regex), but `/openapi.json` documents no response shape at all.

---

## 4. Hardcoded values

None are secrets and none are absolute paths (both greps came back clean). These
are constants that are reasonable today but invisible from `.env`:

| Where | What |
|---|---|
| `config.py:38-42` | `_DEFAULT_MODELS` per provider |
| `chunking.py:28-36` | `SEPARATORS` list |
| `evaluate.py:45-51` | baseline config: chunk 1000 / overlap 0, dense-only |
| `evaluate.py:68-76` | `ABLATIONS`, including the `(0.4, 0.6)` sparse-heavy weights |
| `evaluate.py:326` | sweep grid `[(500,75), (800,120), (1000,150), (1500,225)]` |
| `store.py:76,107` | retry backoff seeded at 2.0s, capped at 60.0s |
| `attribution.py:33-38` | sentence-split and skip-line regexes |
| `prepare_fastapi_docs.py:33-48` | repo URL, docs subdir, skip lists |
| `prepare_fastapi_docs.py:157` | 200-char floor for "this page is a stub" |
| `fetch_local_model.py:17-21` | the three offered GGUF models |
| `Dockerfile:22` | uid 10001, username `orbit` |

---

## 5. Dead or stale code

- **`get_chat_model(streaming: bool = False)`** (`providers.py:47`) — the param
  keys an `lru_cache` but is never passed `True`; `astream` uses the default
  instance. Dead parameter.
- **`ABLATIONS["hybrid+mmr(l=1.0)"]`** — mathematically identical to `"hybrid"`.
  The saved results confirm it: both are 0.8333 / 0.6991 / 0.4667, to four
  decimal places. It measures the same thing twice.
- **`CONFIGS["tuned"]["mmr"]: True`** — a no-op given the 1.0 default; its own
  label says "MMR off". The flag now contradicts the config it produces.
- **`run_ablations` always writes `"detail": []`** — per-question detail is
  collected in `run_config` but silently dropped for ablations, so an ablation
  result cannot be inspected for *which* questions moved.
- **`.dockerignore:11`** — `!data/samples/`, a path that does not exist. The
  sample docs live at `examples/sample-docs/`.
- **`examples/sample-docs/`** — three files referenced by no code anywhere; grep
  for `sample-docs|samples/` across the project returns nothing but the README.
- **`README.md:270-280`** — the `/ask` example cites `samples/api-gateway.md`,
  which does not exist, and asks about a "Standard tier rate limit" that is not
  in the FastAPI corpus. It is an artifact of an earlier corpus.
- **README chunk-count drift** — `README.md:86` says the corpus needs 1,546
  embeddings; `README.md:134` says 1,533 chunks. The saved eval results show both
  numbers historically (1546 in the 20:29 run, 1533 in the 20:33 run).

---

## 6. What only works on this machine

| Thing | Why it does not travel |
|---|---|
| `.env` (untracked) | Sets `LLM_PROVIDER=local`, `CHAT_MODEL`, `LLAMA_N_CTX`, `LLAMA_N_THREADS`, `LLAMA_MAX_TOKENS`. **None of these are in `.env.example`.** This file is the entire reason the project runs here and not on a clean checkout. |
| `.venv` | `torch 2.13.0+cpu` and `llama_cpp 0.3.35`, both installed from custom index URLs that exist only as comments in `requirements.txt`. `pip install -r requirements.txt` reproduces neither. |
| `models/qwen2.5-3b-instruct-q4_k_m.gguf` | 2GB, gitignored. Recoverable via `scripts/fetch_local_model.py`, which is documented — this one is fine. |
| `data/fastapi/` | 141 files from an **unknown** upstream commit (§3.2). |
| `faiss_index/` | Built 2026-08-25 against that unknown corpus. |
| `.embed_cache/models_gemini-embedding-001.jsonl` | Vectors from the abandoned Gemini run. Harmless, but it is a fossil of a provider this project no longer defaults to. |
| `LLAMA_N_THREADS` | Defaults to `os.cpu_count()`, so answer latency is not comparable across machines. The 74.6s measured here is a this-CPU number. |

---

## 7. Open questions

1. **Which FastAPI ref should the corpus pin to?** A release tag makes the corpus
   reproducible forever but freezes it at that version. The current snapshot's
   own commit is unrecoverable, so whatever we pick, today's numbers must be
   re-measured against it.
2. **Confirm the README should carry re-measured numbers.** Fixing §3.3 and
   pinning the corpus will move the published figures. The current ones are not
   reproducible; the new ones will be. I assume you want reproducible over
   flattering, but the ablation story ("MMR hurt, BM25 added nothing") may read
   differently once regenerated, and that is your call to see before I rewrite it.
3. **Is 74.6s per answer acceptable as the shipped default?** It is honest and
   offline, and the README already discloses ~60s. Flagging only because it makes
   any future answer-quality loop impractical to run locally at scale.

### Decisions already made (2026-09-02)

- **Answer-quality scoring is cut from scope.** Retrieval quality is the only
  measured axis. The README will say this explicitly rather than leave the eval
  looking more complete than it is.
- **FAISS stays in-process.** No separate vector-store container; compose brings
  up the app and its index volume. At 1533 chunks a network hop buys nothing.
- **pytest goes in a new `requirements-dev.txt`**, keeping the runtime image slim.
- **Docker mounts the GGUF** rather than baking it; `fetch_local_model.py` becomes
  a documented prerequisite.
