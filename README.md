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
  `[Request Files > File Parameters with UploadFile]`,
  before embedding. A fragment saying "the timeout is 30s" is ambiguous alone
  and unambiguous with its heading attached. Markdown headings give the path
  directly (headings inside code fences are ignored); PDFs get it from their
  bookmark outline. Headings that recur across 3+ documents ("Recap", "Check
  it") are dropped from the breadcrumb (`HEADER_MODE=path-clean`): they say
  nothing about the chunk but are literal text a query can match. On the dev
  set, breadcrumbs beat title-only and no headers on MRR (0.762 vs 0.704 /
  0.732, bge-small, `ablate-20260914-083750.json`) and `path-clean` raised it
  to 0.796. **They are also the main suspect in the held-out result below**:
  every chunk of a page shares its header, so a paraphrased question can land
  on the page's overview chunk instead of the one holding the answer.
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
  misses paraphrases. Both run, and their rankings fuse. The lexical leg is
  SPLADE by default (a learned sparse model that also weights terms a passage
  implies but doesn't contain), with BM25 one env var away. Measured on the
  locked config (`ablate-dev-20260914-155040.json`): hybrid-SPLADE,
  hybrid-BM25 and dense-only all score 32/36 on the dev set; the lexical leg
  only moves ranking (MRR 0.829 / 0.815 / 0.815). Weighting sparse up
  (0.4/0.6) costs a question.
- **MMR re-ranking, measured and then switched off.** It picks candidates that
  are relevant *and* mutually dissimilar. On this corpus sibling chunks of one
  long page are usually *all* relevant, so penalising similarity removed answers
  rather than redundancy. `MMR_LAMBDA` defaults to 1.0 (off). Measured
  twice: at λ=0.5 it cost 2 dev questions and 24 points of precision on
  bge-small, and 28 points of precision on the locked config (58.3% → 30.6%).
- **Cross-encoder re-ranking.** A bi-encoder compares two vectors computed
  independently; a cross-encoder reads question and passage together. With
  `RERANK=true`, the first stage's top 30 are re-scored and the best 5 kept.
  **Measured, and off.** bge-reranker-base alone took the dev set from 32 to
  29/36: it recovered one question and lost four by promoting overview and
  "Recap" chunks. Fusing its rank with the first-stage rank (RRF,
  `RERANK_FUSION`) prevents the losses but also blocks the gain; ColBERT
  (late interaction) and MiniLM behaved the same way. Every re-ranker lowered
  MRR and added 5–35 s per query on CPU (`ablate-20260914-083750.json`,
  `ablate-dev-20260914-131904.json`).
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
| local | `EMBED_PROVIDER=local` | `snowflake-arctic-embed-m` on ONNX Runtime: no key, no quota, CPU only, no PyTorch. Chat can stay on OpenAI. |
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

| 12,000 PDF pages (24 files) | before (`f68f9a25`) | after (`3e6b116e`) |
|---|---|---|
| chunks | 48,481 | 44,517 |
| ingest: load + chunk + embed + index + save | 216.8 s | 75.6 s |
| of which PDF loading | 122.8 s (1 process) | 54.5 s (parallel workers) |
| peak RSS | 4,023 MB | 1,092 MB |
| serving RSS | 1,541 MB | 1,051 MB |
| query latency p50 / p95 | 394 / 506 ms | 80 / 90 ms |
| index on disk | 383 MB | 362 MB |

`bench/results/scale-before-m2-20260914-074137.json` (the original code plus
the benchmark script) and `bench/results/scale-after-20260914-155232.json`.
Same machine (8 cores, 16 GB, Windows), same generated corpus, 1536-d
stand-in vectors, BM25 lexical leg.

What had to change to get there:

- **The embedding cache lived in RAM as Python floats.** The old cache was
  JSONL of float lists; at 48k chunks it peaked at 4 GB. Vectors now live in a
  float32 file read through a memory map, and FAISS is filled in blocks.
- **BM25 was the query bottleneck.** `rank_bm25` scores every document in
  Python for every query term: about 400 ms per query at 48k chunks.
  `app/bm25.py` keeps its exact formula over an inverted index; a test checks
  the scores are bit-identical.
- **PDF handling.** Loading runs in parallel worker processes, running headers
  and page numbers are stripped, and pages are re-joined before splitting, so
  chunks can cross page breaks (and fewer tiny page-tail chunks: 48,481 →
  44,517).
- **SPLADE does not scale on CPU.** The default lexical leg is a transformer
  pass per chunk: 1,855 chunks took 490 s (~3.8 chunks/s, 3.7 GB peak;
  `scale-splade-500-20260914-160124.json`), so roughly 3 hours for 44.5k
  chunks. For a 10k-page corpus on CPU, set `SPARSE=bm25` (what the table
  measures). On the dev set it costs nothing in hit rate (32/36 either way).

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
| `baseline` | fixed-width 1000-char chunks, no overlap, no headers, dense-only top-k, `bge-small-en-v1.5` (the original pipeline's model) |
| `tuned`    | the shipping config: structure-aware chunks + overlap + `path-clean` breadcrumb headers, hybrid SPLADE/dense, `snowflake-arctic-embed-m` |

There are two question sets, with the same schema:

- **dev** (`eval/questions.yaml`, 36 questions): tuned against, freely.
- **held-out** (`eval/questions_heldout.yaml`, 50 questions): pages sampled
  with a fixed seed from pages the dev set never touches, question types
  assigned before each page was read, frozen in commit `b4a42bf3` *before* any
  further tuning, and scored exactly once, after the config was locked.

Results on the locked config (commit `3e6b116e`, local ONNX embeddings, k=5):

| set | config | hit@5 | 95% CI | hit@1 | MRR | exact | paraphrase | multihop |
|---|---|---|---|---|---|---|---|---|
| dev | baseline | 75.0% (27/36) | 59–86% | 53% | 0.625 | 12/12 | 9/12 | 6/12 |
| dev | tuned | **88.9% (32/36)** | 75–96% | 78% | 0.829 | 12/12 | 11/12 | 9/12 |
| **held-out** | baseline | **82.0% (41/50)** | 69–90% | 66% | 0.711 | 17/17 | 9/17 | 15/16 |
| **held-out** | tuned | **78.0% (39/50)** | 65–87% | 52% | 0.632 | 17/17 | 6/17 | 16/16 |
| both (86) | baseline | 79.1% (68/86) | 69–86% | | | | | |
| both (86) | tuned | 82.6% (71/86) | 73–89% | | | | | |

Files: `eval/results/eval-dev-20260914-154938.json` and
`eval/results/eval-heldout-20260914-155003.json`. The numbers use local ONNX
embeddings, because no OpenAI key was available when they were measured;
`python -m eval.evaluate --compare --embed openai:text-embedding-3-small`
reproduces the comparison on OpenAI embeddings.

**The tuning did not generalise.** On the questions it was tuned against,
top-5 relevance rose from 75.0% to 88.9%. On 50 questions it never saw, the
tuned config scored *below* the untuned baseline (78.0% vs 82.0%). The loss
is entirely in paraphrase questions (6/17 vs 9/17); the tuned config gains
one multihop question. The intervals overlap heavily at this sample size, so
the fair summary is "no measurable improvement on unseen questions" rather
than a precise regression.

What the misses show: 5 of the tuned config's 11 held-out misses retrieve the
*right page* but a chunk that lacks the answer. Section-breadcrumb headers
give every chunk of a page the same leading text, so a paraphrased question
matches the page's overview chunk rather than the one holding the code. The
baseline has no headers, so chunk content alone decides. The dev set partly
hid this because its paraphrase questions accept several answer strings,
while most held-out ones accept one identifier. The obvious next experiment
is to keep the header in the lexical leg's text but not in the dense
embedding. Testing it credibly needs a fresh held-out set, since this one has
now been seen.

**`hit_rate@5` is the "top-5 retrieval relevance" number.** Every run saved
with `--save` lands in `eval/results/` with the git commit, the golden set's
hash and the embedding model, so any number here can be traced to the run
that produced it.

### What the ablation showed

Every change tried on the dev set, in order. All are saved in
`eval/results/` except #3 and #4, which were run once as a diagnostic and not
saved.

| # | change (on top of the best so far) | dev hit@5 | MRR | verdict |
|---|---|---|---|---|
| – | breadcrumb headers + hybrid BM25 (bge-small) | 32/36 | 0.762 | starting tuned config |
| 1 | `path-clean` headers | 32/36 | 0.796 | shipped |
| 2 | bge-base-en-v1.5 | 30/36 | 0.812 | no (−2) |
| 3 | deep rank fusion (each leg k=20) | 32/36 | 0.734 | no |
| 4 | bge-small + bge-base + BM25 fusion | 32/36 | 0.764 | no |
| 5–8 | re-rankers: bge-reranker-base + RRF, ColBERT, ColBERT + RRF, MiniLM + RRF | 31–32/36 | 0.687–0.769 | no (5–35 s/query) |
| 9–11 | chunk size 500 / 800 / 1500 | 29 / 32 / 31 | 0.708 / 0.720 / 0.755 | no; 1000 stays |
| 12 | SPLADE lexical leg | 32/36 | 0.819 | rank-only gain |
| 13 | snowflake-arctic-embed-m | 32/36 | 0.815 | 1-for-1 swap |
| 14 | nomic-embed-text-v1.5 | 27/36 | 0.663 | no |
| 15 | arctic-embed-m + SPLADE | 32/36 | 0.829 | **locked** (best ranking) |

None of the 15 changes moved the dev set past 32/36. Of the four remaining dev
misses, two are vocabulary gaps (the static-files page never says "CSS" or
"images", and no retriever put it in its top 50), and two are ranking misses
at ranks 9–24. `python -m eval.evaluate --config tuned --depth 50` prints that
per-leg diagnostic for any run.

The first dev ablation (bge-small, `ablate-20260914-083750.json`) is where the
header, MMR, sparse-weight and first re-ranker results above come from.

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

- **Status:** the script exists, but the service has not been deployed yet.
- **Instance:** serving a 12k-page index with BM25 held about 1.05 GB RSS, and
  loading SPLADE takes a process past 2 GB (`scale-splade-500`). So t3.medium
  (4 GB) with the default `SPARSE=splade`, or t3.small (2 GB) with
  `SPARSE=bm25`.
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
| `HEADER_MODE`    | path-clean | `path-clean` / `path` (section breadcrumb) / `title` / `none`; see the held-out caveat |
| `TOP_K`          | 5       | Chunks handed to the model                        |
| `FETCH_K`        | 20      | Candidates considered before MMR                  |
| `MMR_LAMBDA`     | 1.0     | 1.0 = pure relevance, 0.0 = pure diversity        |
| `USE_HYBRID`     | true    | BM25 + dense ensemble                             |
| `WEIGHT_DENSE`   | 0.6     | Dense weight in the fusion (sparse is 0.4)        |
| `SPARSE`         | splade  | Lexical leg: `splade` or `bm25` (far faster ingest) |
| `RERANK`         | false   | Re-ranking stage (cross-encoder or ColBERT); measured as a loss |
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
  bm25.py              inverted-index BM25, bit-identical to rank_bm25
  splade.py            learned sparse (SPLADE) lexical leg
  retriever.py         hybrid lexical + dense, optional re-rank with rank fusion, citations
  attribution.py       computed citations for models that won't emit them
  rag.py               condense -> retrieve -> ground -> answer
  api.py               FastAPI service
  ui/index.html        streaming chat UI, zero dependencies
eval/
  questions.yaml       dev golden set (36)
  questions_heldout.yaml  held-out set (50), frozen before tuning, scored once
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
