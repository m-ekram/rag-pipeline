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
  is right for corpora with real near-duplicates - just not this one.
- **Contextual chunk headers.** Each chunk is prefixed with `[Doc title - p.12]`
  before embedding. A fragment saying "the timeout is 30s" is ambiguous alone
  and unambiguous with two words of context.
- **Grounding is enforced in the prompt.** The model is told to answer only from
  the numbered passages and to say what's missing rather than fill the gap.
- **Citations are real.** `[2]` maps to an actual chunk with a file path and
  page number, surfaced in the UI and in the API response.
- **Citations survive weak models.** Hosted models follow "cite `[1]` after every
  claim"; a 3B local model does not — measured here, Qwen2.5-3B obeyed a plain
  system instruction ("answer in one word" → "Green") yet emitted zero citations
  across four prompt formulations. So `CITATION_MODE=auto` computes attribution
  instead: each sentence is embedded and matched against the passages actually
  retrieved, and the best match above `CITE_THRESHOLD` becomes its marker. A
  sentence with no confident match gets none, rather than a misleading one.
  Code blocks are never annotated. Set `CITATION_MODE=model` to trust the model
  instead, or `off` to disable.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate
```

```bash
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

### Why local by default

This started on Gemini's free tier and moved off it for a concrete reason:
embedding is capped at **1,000 requests/day**, and the corpus needs 1,546. A
one-time bulk job that cannot finish in a day is not a workable ingest path, and
the eval loop below re-indexes constantly. Locally that cost is zero.

| Provider | Set in `.env` | Notes |
|---|---|---|
| **local** (default) | `LLM_PROVIDER=local` | No key, no quota, offline. ~60 s/answer on CPU. |
| google | `LLM_PROVIDER=google` | Fast (~2-5 s) but 1,000 embeds/day; models retire without notice. |
| openai | `LLM_PROVIDER=openai` | Fast, ~$0.01 to embed this corpus. Needs billing. |

Embeddings and chat are configured independently — `EMBED_PROVIDER` defaults to
`LLM_PROVIDER` but can differ. Local embeddings with a hosted chat model is a
sensible mix: it removes the quota wall that actually bites while keeping fast
answers.

### Add a corpus

Drop your own documents into `data/` — `.pdf`, `.md`, `.rst`, `.txt`, `.html`,
`.docx` are all supported. Three small sample docs live in `examples/sample-docs/`
if you want something to try immediately.

Or build the FastAPI documentation corpus:

```bash
python scripts/prepare_fastapi_docs.py
```

Clones the FastAPI repo (shallow, sparse, MIT-licensed) and flattens
`docs/en/docs` into ~141 clean markdown files in `data/fastapi/`. The script
exists because MkDocs-Material markdown needs real preprocessing before it is
usable as a corpus:

- **Code examples are not in the markdown.** They are `{* ../../docs_src/...  *}`
  include directives pointing at real `.py` files. Left alone, every "how do I
  do X" chunk loses the code that answers it. All 440 are inlined as fenced
  blocks — this is the single biggest quality difference in the corpus.
- **Admonitions** (`/// tip` … `///`, and `////` when nested) become bold
  labels, keeping the text — which is often the most specific content on a page.
- **`{ #heading-anchors }`** are stripped.
- **`release-notes.md` is excluded.** At 694 KB it would be ~40% of the corpus
  and is almost entirely "Fix typo. PR #123 by @user" — retrieval poison.
  Contributor lists and link directories are dropped for the same reason.

### Build the index

```bash
python -m app.ingest
```

1,533 chunks embed locally in a couple of minutes. Vectors are cached in
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

Measured on the FastAPI corpus (36 questions, 141 documents):

```
config        chunks     hit@5       MRR   precision
----------------------------------------------------
baseline        1081     75.0%     0.635      42.8%
tuned           1533     83.3%     0.699      46.7%
----------------------------------------------------
top-5 retrieval relevance: 75% -> 83% (+8%)
```

**`hit_rate@5` is the "top-5 retrieval relevance" number.**

The golden set mixes three failure classes deliberately, because they break for
different reasons and an average over only one of them is misleading:

| class | what it tests | tuned |
|---|---|---|
| `exact` | names an identifier verbatim (`UploadFile`, `root_path`) | 12/12 |
| `paraphrase` | describes the concept with little lexical overlap | 10/12 |
| `multihop` | answer lives in a section the question doesn't name | 8/12 |

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
sparse-heavy         hit@5  33.3%  MRR 0.238  P@5  11.1%
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
  (0.4/0.6) collapses retrieval to 33.3%.

This is the point of the harness: two of the three "obvious" retrieval
improvements were neutral or harmful, and only measurement showed it.

Other modes:

```bash
python -m eval.evaluate --sweep --save
```

`--sweep` grids over chunk sizes (500/800/1000/1500) so you can pick a size on
evidence. `--save` writes JSON to `eval/results/`.

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

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"What is the Standard tier rate limit?"}'
```

```json
{
  "answer": "600 requests per minute, with a burst of 100 [1].",
  "sources": [
    { "n": 1, "source": "samples/api-gateway.md", "page": null, "snippet": "..." }
  ],
  "search_query": "What is the Standard tier rate limit?",
  "latency_ms": 842
}
```

Multi-turn: pass `history` as a list of `{role, content}`. Follow-ups are
rewritten into standalone search queries before retrieval, so "and for
Enterprise?" resolves against the previous turn.

## Docker

```bash
docker compose run --rm ingest
```

```bash
docker compose up --build
```

`data/` mounts read-only and `faiss_index/` mounts read-write — the index is a
volume, not a baked layer, since it's large, changes on a different cadence than
the code, and holds document content.

## Deploying on EC2

```bash
sudo yum install -y docker git && sudo systemctl enable --now docker
```

Then clone, create `.env`, `docker compose run --rm ingest`, `docker compose up -d`.

A `t3.small` handles a corpus of this size; FAISS is in-memory, so size the
instance by index size — roughly `chunks × dimensions × 4 bytes` (10k chunks at
768 dims ≈ 30 MB, comfortable anywhere). Put the container behind nginx or an
ALB for TLS, and set `ALLOWED_ORIGINS` to your actual origin instead of `*`
before exposing it publicly.

## Configuration

Every knob is an env var, documented in `.env.example`. The ones that matter:

| Variable        | Default | Effect                                            |
|-----------------|---------|---------------------------------------------------|
| `CHUNK_SIZE`    | 1000    | Larger = more context per hit, less precise       |
| `CHUNK_OVERLAP` | 150     | Stops answers being severed at a chunk boundary   |
| `TOP_K`         | 5       | Chunks handed to the model                        |
| `FETCH_K`       | 20      | Candidates considered before MMR                  |
| `MMR_LAMBDA`    | 0.5     | 1.0 = pure relevance, 0.0 = pure diversity        |
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

Change any of them, re-run `python -m eval.evaluate`, keep the change if the
number moved the right way.

## Layout

```
config.py              env-driven configuration, provider defaults
app/
  providers.py         chat + embedding factories (google | openai)
  loaders.py           pdf/md/txt/html/docx -> Documents with metadata
  chunking.py          naive (baseline) and structured (shipping) splitters
  store.py             FAISS build / save / load, batched retrying embeds
  retriever.py         hybrid BM25 + dense/MMR, context and citation formatting
  rag.py               condense -> retrieve -> ground -> answer
  api.py               FastAPI service
  ui/index.html        streaming chat UI, zero dependencies
eval/
  questions.yaml       golden set
  evaluate.py          hit_rate / MRR / precision, baseline vs tuned
```

## Notes and limits

- FAISS is loaded fully into memory and rebuilt wholesale on ingest. There's no
  incremental update path; for a corpus that changes often, re-ingest on a
  schedule or move to a server-backed store.
- Conversation history lives in the browser, not on the server. The API is
  stateless — history is passed with each request.
- `langchain-community` (which provides the FAISS wrapper and BM25) emits a
  sunset deprecation warning on import. It works; the retriever already falls
  back across `langchain-classic` and `langchain` import paths for
  `EnsembleRetriever`.
