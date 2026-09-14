# RAG-Based Document Q&A Chatbot

Ask natural-language questions over a corpus of technical documentation and get
answers grounded in the source text, with a citation on every claim.

Python · LangChain · FAISS · OpenAI API · FastAPI · Docker · AWS EC2

- **Answers from the docs, not from memory.** Hybrid retrieval over a FAISS
  index, a grounded prompt, and numbered citations that map to a real file and
  page.
- **Built and load-tested for 10,000+ page corpora.** Parallel PDF extraction,
  page-merged chunking, a float32 on-disk embedding cache, and a benchmark that
  runs the whole pipeline at 12,000 pages ([Scale](#scale-10000-pages)).
- **Retrieval quality is measured, not assumed.** A golden set, baseline vs
  tuned comparison, and per-component ablations, all saved as JSON with the
  commit that produced them ([Measuring retrieval](#measuring-retrieval-quality)).
- **Ships as a container.** Dockerized FastAPI service with a one-command
  EC2 deployment script ([Deploying on EC2](#deploying-on-aws-ec2)).

---

## What it does

```
data/*.pdf,md,txt,html,docx
        │
        ├─ load ─────────► parallel workers; PDFs per page, running headers/footers stripped,
        │                  bookmark outline -> section names
        ├─ chunk ────────► PDF pages re-joined, structure-aware splits,
        │                  section-breadcrumb header on every chunk
        ├─ embed ────────► OpenAI (default) or local ONNX; batched, retrying, float32 disk cache
        └─ index ────────► FAISS (+ chunk sidecar for BM25)
                                │
   question ─► condense ─► hybrid retrieve (BM25 + dense) ─► [cross-encoder rerank] ─► grounded answer + citations
```

Design decisions worth knowing about:

- **Section-breadcrumb headers.** Each chunk is prefixed with where it sits,
  `[tutorial request files > Request Files > File Parameters with UploadFile]`,
  before embedding. A fragment saying "the timeout is 30s" is ambiguous alone
  and unambiguous with its heading attached. Markdown headings give the path
  directly (headings inside code fences are ignored); PDFs get it from their
  bookmark outline. TBD-HEADERS
- **PDF pages are re-joined before splitting.** Extracting one Document per page
  keeps page numbers, but splitting per page severs every answer that runs
  across a page break. Pages are concatenated, split as one text, and each chunk
  is mapped back to the page it starts on (`page`, plus `page_end` when it spans
  two), so citations still point at a real page.
- **Running headers and footers are removed.** Manuals print the chapter title
  and page number on every page. On a 10k-page corpus that is thousands of
  identical fragments matching every query about that chapter. Edge lines that
  recur (digits ignored) on 30%+ of a PDF's pages are dropped.
- **Hybrid retrieval.** The theory: dense search can smooth away exact
  identifiers (error codes, CLI flags, config keys), while BM25 nails those and
  misses paraphrases. Both run, and their rankings fuse. TBD-HYBRID
- **MMR re-ranking, measured and then switched off.** It picks candidates that
  are relevant *and* mutually dissimilar. On this corpus sibling chunks of one
  long page are usually *all* relevant, so penalising similarity removed answers
  rather than redundancy. `MMR_LAMBDA` defaults to 1.0 (off). TBD-MMR
- **Cross-encoder re-ranking.** A bi-encoder compares two vectors computed
  independently; a cross-encoder reads question and passage together. With
  `RERANK=true`, the first stage's top 30 are re-scored and the best 5 kept.
  TBD-RERANK
- **Grounding is enforced in the prompt.** The model is told to answer only from
  the numbered passages and to say what's missing rather than fill the gap.
- **Citations are real.** `[2]` maps to an actual chunk with a file path,
  section and page number, surfaced in the UI and in the API response. With a
  small local model that won't emit markers, `CITATION_MODE=auto` computes them
  by matching each answer sentence to the retrieved passages.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env                                  # then set OPENAI_API_KEY
```

Build the FastAPI documentation corpus (or drop your own `.pdf`, `.md`,
`.rst`, `.txt`, `.html`, `.docx` files into `data/`):

```bash
python scripts/prepare_fastapi_docs.py
```

Build the index, then run the server and open <http://localhost:8000>:

```bash
python -m app.ingest
uvicorn app.api:app --reload
```

`python -m app.ingest --dry-run` loads and chunks without embedding, and prints
the file, PDF-page and chunk counts. Vectors are cached in `.embed_cache/`, so
re-running after a chunking change only embeds the chunks that changed.

### Providers

| Provider | Set in `.env` | Notes |
|---|---|---|
| **openai** (default) | `LLM_PROVIDER=openai` | `gpt-4o-mini` + `text-embedding-3-small`. Embedding this corpus costs about a cent. |
| local | `EMBED_PROVIDER=local` | `bge-small-en-v1.5` on ONNX Runtime: no key, no quota, CPU only, no PyTorch. Chat can stay on OpenAI. |
| local chat | `LLM_PROVIDER=local` | Qwen2.5-3B on llama.cpp. `pip install -r requirements-local-llm.txt`, then `python scripts/fetch_local_model.py`. About 60 s per answer on CPU. |
| google | `LLM_PROVIDER=google` | Gemini. The free tier caps embedding at 1,000 requests/day. |

Embeddings and chat are configured independently. **Re-run ingestion after
changing the embedding model**: vectors from different models are not
comparable, and `load_index` refuses to open a mismatched index rather than
silently return garbage.

### About the corpus

`scripts/prepare_fastapi_docs.py` clones the FastAPI repo (shallow, sparse,
MIT-licensed) and flattens `docs/en/docs` into 141 clean markdown files. The
MkDocs-Material source needs real preprocessing first:

- **Code examples are not in the markdown.** They are `{* ../../docs_src/... *}`
  include directives. All ~440 are inlined as fenced blocks, which is the single
  biggest quality difference in the corpus.
- **Admonitions** (`/// tip` … `///`) become bold labels, keeping the text.
- **`release-notes.md` is excluded**: 694 KB of "Fix typo. PR #123", about 40% of
  the corpus and pure retrieval noise.

## Scale: 10,000+ pages

The evaluation corpus is deliberately small so the eval loop stays fast. The
pipeline is built for a much larger one, and `bench/scale_test.py` proves it:

```bash
pip install -r requirements-dev.txt
python -m bench.scale_test                 # 12,000 PDF pages
python -m bench.scale_test --embedding openai   # same, with real embeddings (~$0.15)
```

It generates 12,000 pages of technical text as real multi-page PDFs (running
headers, page numbers, a bookmark outline), then runs the production code path:
`app.loaders` extraction, chunking, the embedding cache, FAISS build/save/load,
BM25, and the same hybrid retriever the API serves, followed by 200 timed
queries. By default the vectors are deterministic 1536-d stand-ins (the size of
`text-embedding-3-small`), so memory and index behaviour match a real run while
embedding costs nothing. Page text is sampled from the FastAPI docs, so token
statistics look like documentation.

TBD-SCALE-TABLE

What had to change to get there:

TBD-SCALE-FINDINGS

**Sizing.** FAISS here is a flat (exact) index held in memory:
`chunks × dimensions × 4 bytes`. 10,000 pages is roughly 40k chunks, about
245 MB at 1536 dimensions, with millisecond search. Flat stays the right choice
into the low millions of vectors. Past that, swap in an HNSW or IVF index.

## Measuring retrieval quality

Retrieval quality is the ceiling on answer quality: if the right passage never
reaches the model, no amount of prompt work recovers it. So it's measured
rather than eyeballed.

`eval/questions.yaml` holds a golden set: the question, the file(s) that
should be retrieved, and optionally a string the chunk must contain (so "right
file" alone doesn't score a hit). The 36 questions mix three failure classes,
because they break for different reasons:

| class | what it tests |
|---|---|
| `exact` | names an identifier verbatim (`UploadFile`, `root_path`) |
| `paraphrase` | describes the concept with little lexical overlap |
| `multihop` | answer lives in a section the question doesn't name |

```bash
python -m eval.evaluate --validate          # every question reachable? (no embedding)
python -m eval.evaluate --compare --save    # baseline vs tuned
python -m eval.evaluate --ablate --save     # one component at a time
python -m eval.evaluate --compare --embed openai:text-embedding-3-large
```

| config     | what it is |
|------------|------------|
| `baseline` | fixed-width 1000-char chunks, no overlap, no headers, dense-only top-k |
| `tuned`    | the shipping config: structure-aware chunks + overlap + section-breadcrumb headers, hybrid BM25/dense TBD-TUNED-LABEL |

TBD-EVAL-TABLE

**`hit_rate@5` is the "top-5 retrieval relevance" number.** Every run saved
with `--save` lands in `eval/results/` with the git commit, the golden set's
hash and the embedding model, so any number here can be traced to the run
that produced it.

### What the ablation showed

TBD-ABLATION

This is the point of the harness: an "obvious" retrieval improvement is often
neutral or harmful on a given corpus, and only measurement shows which.

## API

| Method | Path          | Purpose                                      |
|--------|---------------|----------------------------------------------|
| GET    | `/`           | Chat UI                                      |
| GET    | `/health`     | Readiness, active models, index + corpus metadata |
| GET    | `/sources`    | Indexed documents and chunk counts           |
| POST   | `/ask`        | JSON answer with citations                   |
| POST   | `/ask/stream` | Same, streamed as server-sent events         |
| POST   | `/reload`     | Re-open the index after a re-ingest          |
| GET    | `/docs`       | OpenAPI docs                                 |

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"How do I receive an uploaded file?"}'
```

```json
{
  "answer": "Declare a parameter of type UploadFile with File() [1] ...",
  "sources": [
    { "n": 1, "source": "fastapi/tutorial__request-files.md",
      "section": "Request Files > File Parameters with UploadFile", "page": null, "snippet": "..." }
  ],
  "search_query": "How do I receive an uploaded file?",
  "latency_ms": 1840
}
```

Multi-turn: pass `history` as a list of `{role, content}`. Follow-ups are
rewritten into standalone search queries before retrieval, so "and for
multiple files?" resolves against the previous turn.

## Docker

```bash
docker compose run --rm ingest     # build the index
docker compose up --build -d       # serve on :8000
```

The image has no PyTorch: local embeddings and re-ranking run on ONNX Runtime,
which keeps it small enough for a modest instance. `data/` mounts read-only and
`faiss_index/` read-write. The index is a volume, not a baked layer, since it's
large, changes on a different cadence than the code, and holds document
content. A named `cache` volume keeps downloaded models and paid-for embeddings
across restarts.

## Deploying on AWS EC2

`deploy/ec2/setup.sh` takes a fresh Amazon Linux 2023 instance to a running
service: it installs Docker and the Compose plugin, adds swap on small
instances, clones the repo, writes `.env`, builds the corpus and index, and
starts the container on port 80.

```bash
# on the instance, as ec2-user
curl -fsSL https://raw.githubusercontent.com/m-ekram/rag-pipeline/main/deploy/ec2/setup.sh -o setup.sh
OPENAI_API_KEY=sk-... bash setup.sh
```

- **Instance:** TBD-EC2-SIZING
- **Security group:** allow inbound TCP 80 (and 22 from your IP only).
- **Before exposing it publicly:** put it behind an ALB or nginx with TLS, and
  set `ALLOWED_ORIGINS` in `.env` to your real origin instead of `*`.
- **Updating:** re-run the script. It pulls, rebuilds, and restarts; the index is
  rebuilt only if missing (`sudo docker compose run --rm ingest` forces it).

## Configuration

Every knob is an env var, documented in `.env.example`. The ones that matter:

| Variable         | Default | Effect                                            |
|------------------|---------|---------------------------------------------------|
| `CHUNK_SIZE`     | 1000    | Larger = more context per hit, less precise       |
| `CHUNK_OVERLAP`  | 150     | Stops answers being severed at a chunk boundary   |
| `HEADER_MODE`    | TBD-HEADER-DEFAULT | `path` (section breadcrumb) / `title` / `none` |
| `TOP_K`          | 5       | Chunks handed to the model                        |
| `FETCH_K`        | 20      | Candidates considered before MMR                  |
| `MMR_LAMBDA`     | 1.0     | 1.0 = pure relevance, 0.0 = pure diversity        |
| `USE_HYBRID`     | true    | BM25 + dense ensemble                             |
| `WEIGHT_DENSE`   | 0.6     | Dense weight in the fusion (sparse is 0.4)        |
| `RERANK`         | TBD-RERANK-DEFAULT | Cross-encoder second stage              |
| `RERANK_MODEL`   | `BAAI/bge-reranker-base` | Any fastembed cross-encoder      |
| `INGEST_WORKERS` | 0 (auto) | Loader processes; parallel only for large corpora |
| `EMBED_RPM`      | 0 (90 for google) | Client-side embedding rate cap           |
| `CITATION_MODE`  | model   | `model` / `auto` / `off`; `auto` for local chat   |

Change one, re-run `python -m eval.evaluate`, and keep the change if the number
moved the right way.

## Layout

```
config.py              env-driven configuration, provider defaults
app/
  providers.py         chat, embedding and cross-encoder factories (openai | local | google)
  loaders.py           parallel pdf/md/txt/html/docx loading, PDF boilerplate + outline
  chunking.py          naive (baseline) and structured (shipping) splitters, breadcrumbs
  store.py             float32 embedding cache, FAISS build / save / load
  retriever.py         hybrid BM25 + dense, optional cross-encoder rerank, citations
  attribution.py       computed citations for models that won't emit them
  rag.py               condense -> retrieve -> ground -> answer
  api.py               FastAPI service
  ui/index.html        streaming chat UI, zero dependencies
eval/
  questions.yaml       golden set
  evaluate.py          hit_rate / MRR / precision, baseline vs tuned, ablations
  results/             saved runs quoted in this README
bench/
  scale_test.py        10k+ page load test
  results/             saved runs quoted in this README
deploy/ec2/setup.sh    Amazon Linux 2023 -> running service
tests/                 unit tests for cache, chunking and PDF handling
```

## Notes and limits

- FAISS is loaded fully into memory and rebuilt wholesale on ingest (the
  embedding cache makes rebuilds cheap, but there is no incremental index
  update). For a corpus that changes constantly, move to a server-backed store.
- Conversation history lives in the browser, not on the server. The API is
  stateless: history is passed with each request.
- `langchain-community` (FAISS wrapper, BM25) emits a sunset deprecation
  warning on import. It works; the classic retrievers come from
  `langchain-classic`, and a missing install fails loudly instead of silently
  degrading to dense-only.
