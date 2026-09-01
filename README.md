# RAG-Based Document Q&A Chatbot

Ask natural-language questions over a corpus of technical documentation and get
answers grounded in the source text, with a citation on every claim.

Python · LangChain · FAISS · FastAPI · Docker

**Runs fully offline by default** — local embeddings (`bge-small-en-v1.5`) and a
local chat model (`Qwen2.5-3B-Instruct`, Q4_K_M via llama.cpp). No API key, no
quota, no network. Gemini and OpenAI remain available as one-line swaps.

---

## What it does

```
data/*.pdf,md,txt,html,docx
        │
        ├─ load ─────────► per-page / per-file Documents with source metadata
        ├─ chunk ────────► structure-aware splits + contextual headers
        ├─ embed ────────► batched, retrying, provider-agnostic
        └─ index ────────► FAISS (+ chunk sidecar for BM25)
                                │
   question ─► condense ─► hybrid retrieve (BM25 + dense/MMR) ─► grounded answer + citations
```

Design decisions worth knowing about:

- **Hybrid retrieval.** The theory: dense search misses exact identifiers —
  error codes, CLI flags, config keys — because embeddings smooth them away,
  while BM25 nails those and misses paraphrases. Both run and their rankings
  fuse. The measurement: on this corpus BM25 adds nothing, because a modern
  embedding model already scores 12/12 on exact identifiers. See the ablation
  below — the honest version of this bullet is "it's free insurance", not "it's
  why retrieval works".
- **MMR re-ranking, measured and then switched off.** It pulls `FETCH_K`
  candidates and selects `TOP_K` that are relevant *and* mutually dissimilar.
  On this corpus that cost 5.5 points of hit@5 and 20 points of precision, so
  `MMR_LAMBDA` defaults to 1.0 (off). Kept and documented because the mechanism
  is right for corpora with real near-duplicates — just not this one.
- **Contextual chunk headers.** Each chunk is prefixed with `[Doc title - p.12]`
  before embedding. A fragment saying "the timeout is 30s" is ambiguous alone
  and unambiguous with two words of context.
- **Grounding is enforced in the prompt.** The model is told to answer only from
  the numbered passages and to say what's missing rather than fill the gap.
- **Citations are real.** `[2]` maps to an actual chunk with a file path and
  page number, surfaced in the UI and in the API response.
- **Citations survive weak models.** Hosted models follow "cite `[1]` after every
  claim"; a 3B local model does not — measured here, Qwen2.5-3B emitted zero
  citations across four prompt formulations, and still does today when
  `CITATION_MODE=off`. So `CITATION_MODE=auto` computes attribution instead:
  each sentence is embedded and matched against the passages actually
  retrieved, and the best match above `CITE_THRESHOLD` becomes its marker. A
  sentence with no confident match gets none, rather than a misleading one.
  A computed citation is a weaker claim than a model-asserted one — it means
  "this sentence is closest to passage N, above a confidence floor", not "the
  model cited N". Code blocks are never annotated. Set `CITATION_MODE=model` to
  trust the model instead, or `off` to disable.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
```

Two dependencies need a specific wheel index. Install them first, then the rest:

```bash
# CPU-only torch. Without this pip resolves the CUDA build and pulls GBs of
# nvidia wheels that never get used.
pip install --index-url https://download.pytorch.org/whl/cpu torch

# llama-cpp-python publishes no wheel on PyPI; a plain install triggers a source
# build (which also fails on Windows via MAX_PATH).
pip install llama-cpp-python --only-binary :all: \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

pip install -r requirements.txt
```

Copy `.env.example` to `.env`. The defaults are fully local, so no key is
needed:

```bash
cp .env.example .env
```

Fetch the local chat model (~2 GB GGUF; the embedding model downloads itself on
first use):

```bash
python scripts/fetch_local_model.py
```

### Try it in five minutes

Three small sample documents ship in `examples/sample-docs/`. Point the pipeline
at them for a fast end-to-end check before committing to a real corpus:

```bash
DATA_DIR=examples/sample-docs INDEX_DIR=samples_index python -m app.ingest --rebuild
DATA_DIR=examples/sample-docs INDEX_DIR=samples_index uvicorn app.api:app
```

13 chunks, about 10 seconds to embed. Asking "What is the Standard tier rate
limit?" returns:

> The Standard tier rate limit is 600 requests per minute [1].

with `[1]` resolving to the rate-limiting table in `api-gateway.md`.

### Why local by default

This started on Gemini's free tier and moved off it for a concrete reason:
embedding is capped at **1,000 requests/day**, and the corpus needs 1,534. A
one-time bulk job that cannot finish in a day is not a workable ingest path, and
the eval loop below re-indexes constantly. Locally that cost is zero.

| Provider | Set in `.env` | Notes |
|---|---|---|
| **local** (default) | `LLM_PROVIDER=local` | No key, no quota, offline. ~50-75 s/answer on CPU. |
| google | `LLM_PROVIDER=google` | Fast (~2-5 s) but 1,000 embeds/day; models retire without notice. |
| openai | `LLM_PROVIDER=openai` | Fast, ~$0.01 to embed this corpus. Needs billing. |

Embeddings and chat are configured independently — `EMBED_PROVIDER` defaults to
`LLM_PROVIDER` but can differ. Local embeddings with a hosted chat model is a
sensible mix: it removes the quota wall that actually bites while keeping fast
answers.

## Build the documentation corpus

Drop your own documents into `data/` — `.pdf`, `.md`, `.rst`, `.txt`, `.html`,
`.docx` are all supported.

Or build the FastAPI documentation corpus this project is measured on:

```bash
python scripts/prepare_fastapi_docs.py
```

Clones the FastAPI repo (shallow, sparse, MIT-licensed) at a **pinned ref** and
flattens `docs/en/docs` into 141 clean markdown files in `data/fastapi/`. The
script exists because MkDocs-Material markdown needs real preprocessing before
it is usable as a corpus:

- **Code examples are not in the markdown.** They are `{* ../../docs_src/... *}`
  include directives pointing at real `.py` files. Left alone, every "how do I
  do X" chunk loses the code that answers it. 439 of the 440 are inlined as
  fenced blocks — this is the single biggest quality difference in the corpus.
  (One include points at a file absent at this ref; the script reports it.)
- **Admonitions** (`/// tip` … `///`, and `////` when nested) become bold
  labels, keeping the text — which is often the most specific content on a page.
- **`{ #heading-anchors }`** are stripped.
- **`release-notes.md` is excluded.** At 694 KB it would be ~40% of the corpus
  and is almost entirely "Fix typo. PR #123 by @user" — retrieval poison.
  Contributor lists and link directories are dropped for the same reason.

### Reproducibility

The corpus is a build artifact, so it is not committed — but the *recipe* has to
be deterministic or the measured numbers stop describing the corpus they were
measured on. Two things make it so:

- The upstream ref is pinned (`--ref`, default `0.141.1`). Cloning the default
  branch means the corpus silently drifts with upstream.
- Every generated file is digested into `data/fastapi/CORPUS.lock.json`, which
  records the ref and commit and **is** committed.

```bash
python scripts/prepare_fastapi_docs.py --verify
```

```
Corpus matches CORPUS.lock.json: 141 files, ref 0.141.1 @ 95f8322ee1dc
```

Two consecutive `--clean` rebuilds produce byte-identical locks. To move to a
newer FastAPI, bump `--ref`, re-run the eval, and update the checked-in baseline.

## Build the index

```bash
python -m app.ingest
```

1,534 chunks embed locally in a couple of minutes. Vectors are cached in
`.embed_cache/`, so re-running after a chunking change only embeds what actually
changed.

Run the server and open <http://localhost:8000>:

```bash
uvicorn app.api:app --reload
```

## Switching to OpenAI

One env var, no code change:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Defaults become `gpt-4o-mini` and `text-embedding-3-small`. **Re-run ingestion
after switching** — vectors from different embedding models are not comparable,
and `load_index` refuses to open an index built with a different model rather
than silently returning garbage.

## Measuring retrieval quality

Retrieval quality is the ceiling on answer quality: if the right passage never
reaches the model, no amount of prompt work recovers it. So it's measured
rather than eyeballed.

`eval/questions.yaml` holds a golden set — question, the file(s) that should be
retrieved, and optionally a string the chunk must contain (so "right file" alone
doesn't score a hit).

```bash
python -m eval.evaluate --compare
```

This builds two indexes and reports both:

| config     | what it is                                                        |
|------------|-------------------------------------------------------------------|
| `baseline` | fixed-width chunks, no overlap, dense-only top-k                   |
| `tuned`    | structure-aware chunks + overlap + headers, hybrid BM25/dense, MMR off |

Measured on the FastAPI corpus at ref `0.141.1` (36 questions, 141 documents):

```
config                chunks     hit@5       MRR   precision
------------------------------------------------------------
baseline                1081    75.0%     0.635      42.8%
tuned                   1534    83.3%     0.699      46.7%
------------------------------------------------------------
top-5 retrieval relevance: 75% -> 83% (+8%)
```

**`hit_rate@5` is the "top-5 retrieval relevance" number.**

The golden set mixes three failure classes deliberately, because they break for
different reasons and an average over only one of them is misleading:

| class | what it tests | baseline | tuned |
|---|---|---|---|
| `exact` | names an identifier verbatim (`UploadFile`, `root_path`) | 12/12 | 12/12 |
| `paraphrase` | describes the concept with little lexical overlap | 8/12 | 10/12 |
| `multihop` | answer lives in a section the question doesn't name | 7/12 | 8/12 |

Every remaining miss is `paraphrase` or `multihop`. Exact-identifier retrieval
is solved; the headroom is in questions whose wording shares nothing with the
source text.

Before spending time on embeddings, check the set is even answerable:

```bash
python -m eval.evaluate --validate
```

This catches questions whose `relevant_sources` don't exist or whose
`must_contain` string appears in no chunk — those can never score a hit and
would silently drag the number down.

### Guarding against regressions

`eval/baseline.json` is committed, and records the corpus ref and commit the
numbers were measured on:

```bash
python -m eval.evaluate --check
```

Re-runs the eval and exits non-zero if any metric dropped more than
`--tolerance` (default 0.02) below the baseline. Retrieval here is
deterministic — a clean re-run reports deltas of exactly `+0.000` — so the
tolerance is slack for a corpus or library bump, not for noise. Write a new
baseline with `--save-baseline` once you have decided a change is an
improvement.

### What the ablation actually showed

```bash
python -m eval.evaluate --ablate
```

Varies only the retriever, reusing one embedded index, so each component is
isolated:

```
dense-only           hit@5  83.3%  MRR 0.705  P@5  47.2%
dense+mmr            hit@5  77.8%  MRR 0.675  P@5  27.2%
hybrid               hit@5  83.3%  MRR 0.699  P@5  46.7%
hybrid+mmr           hit@5  77.8%  MRR 0.648  P@5  27.2%
hybrid+mmr(l=0.8)    hit@5  77.8%  MRR 0.664  P@5  38.3%
sparse-heavy         hit@5  36.1%  MRR 0.236  P@5  11.7%
```

Two findings, both of which contradicted the original design:

- **MMR was hurting.** At `lambda=0.5` it cost 5.5 points of hit@5 and 20 points
  of precision. Sibling chunks of one long page are usually *all* relevant, so
  penalising similarity removed answers rather than redundancy. `MMR_LAMBDA` now
  defaults to 1.0 (off). Lower it only for corpora with genuine near-duplicates.
- **Hybrid BM25 contributes nothing here** — dense-only and hybrid score
  identically, with the same six misses. `bge-small-en-v1.5` already gets 12/12
  on exact identifiers unaided, which is precisely the job BM25 was added to do.
  It stays enabled since it costs nothing and helps corpora with rarer tokens,
  but on this corpus it is not what earns the score. Weighting sparse *up*
  (0.4/0.6) collapses retrieval to 36.1%.

This is the point of the harness: two of the three "obvious" retrieval
improvements were neutral or harmful, and only measurement showed it.

Other modes:

```bash
python -m eval.evaluate --sweep --save
```

`--sweep` grids over chunk sizes (500/800/1000/1500) so you can pick a size on
evidence. `--save` writes JSON to `eval/results/`.

### What is not measured

**Answer quality is not scored.** The eval measures retrieval only — whether the
right passage reached the model, and how highly it ranked. Whether the model
then wrote a good answer from it is currently judged by reading, not by a
metric. Treat every number above as a statement about retrieval, and nothing
more.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

215 tests, offline and hermetic — no index is loaded, no model weights are read,
and config tests stub `load_dotenv` so they assert shipped defaults rather than
whatever is in your `.env`. The whole suite runs in about 15 seconds.

## API

| Method | Path          | Purpose                                      |
|--------|---------------|----------------------------------------------|
| GET    | `/`           | Chat UI                                      |
| GET    | `/health`     | Readiness, active models, index metadata     |
| GET    | `/sources`    | Indexed documents and chunk counts           |
| POST   | `/ask`        | JSON answer with citations                   |
| POST   | `/ask/stream` | Same, streamed as server-sent events         |
| POST   | `/reload`     | Re-open the index after a re-ingest          |
| GET    | `/docs`       | OpenAPI docs                                 |

Every endpoint publishes a request *and* response schema, so `/openapi.json` is
enough to generate a client.

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"How do I return a custom status code from a path operation?"}'
```

```json
{
  "answer": "To return a custom status code from a path operation, you can use the `status_code` parameter in your path operation function [2].",
  "sources": [
    {
      "n": 2,
      "source": "fastapi/tutorial__path-operation-configuration.md",
      "title": "tutorial  path operation configuration",
      "page": null,
      "chunk_id": "889afb82a0117892",
      "snippet": "[tutorial  path operation configuration]
## Response Status Code
You can define the (HTTP) `status_code` to be used in the response of your *path operation*..."
    }
  ],
  "search_query": "How do I return a custom status code from a path operation?",
  "latency_ms": 66568
}
```

Multi-turn: pass `history` as a list of `{role, content}`. Follow-ups are
rewritten into standalone search queries before retrieval, so "and for
Enterprise?" resolves against the previous turn.

## Docker

FAISS runs in-process rather than as a service, so what compose brings up is the
app plus the volumes holding its index and model weights. At this corpus size a
network hop to a separate vector store would add operational surface and buy
nothing.

Fetch the chat model on the host first — it is mounted read-only rather than
baked into the image, which would add ~2 GB to every build:

```bash
python scripts/fetch_local_model.py
```

```bash
docker compose run --rm ingest
```

```bash
docker compose up
```

The index, the HuggingFace cache and the embed cache are named volumes rather
than bind mounts: the container runs as uid 10001, and a bind mount would hand
it a host-owned directory it cannot write. `data/` and `models/` mount
read-only.

Measured on this machine: the image builds to 2.48 GB, ingestion embeds 1,534
chunks in 286 s, and `/ask` answers in ~67 s.

## Deploying on EC2

```bash
sudo yum install -y docker git && sudo systemctl enable --now docker
```

Then clone, create `.env`, fetch the model, `docker compose run --rm ingest`,
`docker compose up -d`.

A `t3.small` handles a corpus of this size *for retrieval*; local chat inference
wants more CPU than that. FAISS is in-memory, so size the instance by index size
— roughly `chunks × dimensions × 4 bytes` (10k chunks at 768 dims ≈ 30 MB,
comfortable anywhere). Put the container behind nginx or an ALB for TLS, and set
`ALLOWED_ORIGINS` to your actual origin instead of `*` before exposing it
publicly.

## Configuration

Every knob is an env var, documented in `.env.example`. No secrets and no
absolute paths live in the code. The ones that matter:

| Variable        | Default | Effect                                            |
|-----------------|---------|---------------------------------------------------|
| `CHUNK_SIZE`    | 1000    | Larger = more context per hit, less precise       |
| `CHUNK_OVERLAP` | 150     | Stops answers being severed at a chunk boundary   |
| `TOP_K`         | 5       | Chunks handed to the model                        |
| `FETCH_K`       | 20      | Candidates considered before MMR                  |
| `MMR_LAMBDA`    | 1.0     | 1.0 = pure relevance (off), 0.0 = pure diversity  |
| `USE_HYBRID`    | true    | BM25 + dense ensemble                             |
| `WEIGHT_DENSE`  | 0.6     | Dense weight in the fusion (sparse is 0.4)        |
| `EMBED_RPM`     | 0 local | Client-side embedding rate cap; 0 = off (see below) |
| `CITATION_MODE` | auto    | `model` / `auto` / `off` (see Citations above)      |
| `CITE_THRESHOLD`| 0.55    | Similarity floor for computed attribution          |
| `LLAMA_N_CTX`   | 4096    | Local context window; must hold TOP_K passages     |
| `LLAMA_N_THREADS`| cores  | llama.cpp worker threads                           |

**On `EMBED_RPM`:** the provider counts one request per *document*, not per
batch, and Google's free tier caps embedding at 100/min. A 1,500-chunk corpus is
therefore ~17 minutes of wall clock, paced. Ingestion throttles itself to stay
under the ceiling rather than sprinting into a 429 and burning retries — raise
it if you are on a paid tier.

Change any of them, re-run `python -m eval.evaluate --check`, keep the change if
the number moved the right way.

## Layout

```
config.py              env-driven configuration, provider defaults
app/
  providers.py         chat + embedding factories (local | google | openai)
  loaders.py           pdf/md/txt/html/docx -> Documents with metadata
  chunking.py          naive (baseline) and structured (shipping) splitters
  store.py             FAISS build / save / load, batched retrying embeds
  retriever.py         hybrid BM25 + dense/MMR, context and citation formatting
  attribution.py       computed citations for models that will not emit them
  rag.py               condense -> retrieve -> ground -> answer
  api.py               FastAPI service, request and response schemas
  ui/index.html        streaming chat UI, zero dependencies
eval/
  questions.yaml       golden set
  evaluate.py          hit_rate / MRR / precision, baseline vs tuned, ablations
  baseline.json        committed numbers, with the corpus ref they came from
scripts/
  prepare_fastapi_docs.py   pinned corpus builder + lock file
  fetch_local_model.py      GGUF downloader
tests/                 215 offline tests
```

## Notes and limits

- **Answer quality is not measured** — see above. Retrieval is.
- Local answer latency (~50-75 s) is a property of this CPU and `LLAMA_N_THREADS`,
  not a portable number.
- FAISS is loaded fully into memory and rebuilt wholesale on ingest. There's no
  incremental update path; for a corpus that changes often, re-ingest on a
  schedule or move to a server-backed store.
- Conversation history lives in the browser, not on the server. The API is
  stateless — history is passed with each request.
- `langchain-community` (which provides the FAISS wrapper and BM25) emits a
  sunset deprecation warning on import. It works; the retriever already falls
  back across `langchain-classic` and `langchain` import paths for
  `EnsembleRetriever`.
