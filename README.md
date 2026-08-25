# RAG-Based Document Q&A Chatbot

Ask natural-language questions over a corpus of technical documentation and get
answers grounded in the source text, with a citation on every claim.

Python · LangChain · FAISS · FastAPI · Docker

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

- **Hybrid retrieval.** Dense-only search misses exact identifiers — error
  codes, CLI flags, config keys — because embeddings smooth them away. BM25
  nails those and misses paraphrases. Both run, and their rankings are fused.
- **MMR re-ranking.** Pulls `FETCH_K` candidates and selects `TOP_K` that are
  relevant *and* mutually dissimilar, so all five slots aren't near-duplicate
  paragraphs from one page.
- **Contextual chunk headers.** Each chunk is prefixed with `[Doc title - p.12]`
  before embedding. A fragment saying "the timeout is 30s" is ambiguous alone
  and unambiguous with two words of context.
- **Grounding is enforced in the prompt.** The model is told to answer only from
  the numbered passages and to say what's missing rather than fill the gap.
- **Citations are real.** `[2]` maps to an actual chunk with a file path and
  page number, surfaced in the UI and in the API response.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add a key. The default provider is Google
(free tier — get a key at <https://aistudio.google.com/apikey>):

```bash
cp .env.example .env
```

Drop documents into `data/` (three sample docs ship in `data/samples/`), then
build the index:

### Or use the FastAPI docs as a corpus

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

```bash
python -m app.ingest
```

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
| `tuned`    | structure-aware chunks + overlap + headers, hybrid BM25/dense, MMR |

Output:

```
config        chunks    hit@5       MRR   precision
--------------------------------------------------
baseline           9    ....%     ....       ....%
tuned             13    ....%     ....       ....%
--------------------------------------------------
top-5 retrieval relevance: ..% -> ..%
```

**`hit_rate@5` is the "top-5 retrieval relevance" number.** Run it on your own
corpus and golden set — the figure that matters is the one you measure, and the
20 shipped questions over 3 sample docs are a harness demo, not a benchmark. On
a real corpus, aim for 30–50 questions mixing paraphrases, exact identifiers,
and multi-hop questions; those three classes fail for different reasons, and
mixing them is what makes the number meaningful.

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
| `EMBED_RPM`     | 90      | Client-side embedding rate cap (see below)        |

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
