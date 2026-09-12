# 3. System Architecture

## 3.1 Overview

Sanchay is a single-user system that runs on the user's machine. A browser
interface (Next.js) talks to a Python backend (FastAPI), which owns the whole
pipeline: extraction, indexing, retrieval and generation. Only the answer
engine may be remote.

![Component architecture of Sanchay](../figures/fig_architecture.png)

The backend runs all pipeline work on a single dedicated thread. Some stores
cannot cross threads (SQLite), and on a four-core laptop concurrent OCR and
embedding contend for the CPU rather than parallelise. Serialising work is
acceptable for one user, provided the interface is told when a request is
waiting — a requirement that became central to the optimisation work.

## 3.2 Components

| Layer | Responsibility |
|---|---|
| Web interface | Folder picker, engine picker, indexing progress, streamed answers with citations and per-stage timings |
| API | Streams newline-delimited JSON for indexing and chat; heartbeats; cancellation on disconnect; model warm-up; serves the built interface |
| Extraction | Native text first; layout-aware table recovery; OCR only for weak pages; a content-addressed per-page cache |
| Chunking | Parent/child units per structure: table rows, prose sections, voter cards grouped by household |
| Lexical index | SQLite FTS5 with BM25, slash-safe tokens for IDs, field search, OCR-tolerant ID matching |
| Dense index | multilingual-e5-small embeddings in a persistent NumPy index (Qdrant when a server is available) |
| Intent router | Classifies each question and chooses a structured lookup or hybrid retrieval |
| Reranker | Cross-encoder for semantic questions only |
| Generation | Threshold gate, prompt packing within a token budget, streaming answer engine, citation and fact checks |

## 3.3 Indexing flow

1. The API scans the chosen folder recursively (skipping caches and virtual
   environments) and detects the OCR language from file names (Hindi rolls
   are named `…-HIN-…`).
2. One extractor processes every file, so the OCR model loads once. Each page
   is served from the cache when possible; otherwise native text is scored
   for quality and kept if good, and OCR runs only when it is not. Tables that
   continue across pages are stitched.
3. Pages are chunked by their own structure. The FTS5 index is built in
   memory in seconds.
4. The chunk texts are fingerprinted. If the stored vectors match, they are
   reused; otherwise the chunks are embedded.
5. The reranker is chosen from the share of Devanagari and Arabic letters, the
   answer engine is resolved, and the local model is loaded into memory.

## 3.4 Question flow

1. The router classifies the question: page lookup, voter ID, serial number,
   relation, house, administrative summary, list question, or general
   semantic question.
2. Structured intents use direct index lookups and are never reranked; their
   relevance is exact. Semantic questions use Reciprocal Rank Fusion of dense
   and lexical results followed by cross-encoder reranking.
3. The pipeline sizes the evidence to the question: list questions keep up to
   30 records; others keep the top few.
4. A threshold gate can abstain before any model call.
5. The prompt numbers the evidence, labels it with document and page when it
   spans documents, and fits it into a token budget chosen for the engine.
6. The model's tokens stream to the browser. Citations are validated and
   expanded into document/section/page references; numbers are checked
   against the evidence.

## 3.5 Streaming protocol

Both long operations stream newline-delimited JSON over a POST body. Event
types are `progress`, `status`, `heartbeat`, `token`, `done` and `error`. A
heartbeat carrying the current stage and elapsed time is sent every two
seconds whenever the pipeline is silent, so that neither the user nor any
proxy mistakes a long OCR pass or model load for a dead connection. Closing
the stream cancels the request inside the worker.

## 3.6 Deployment modes

In development the Next.js dev server proxies `/api/*` to FastAPI. In
production the interface is exported as static files and served by FastAPI
itself: one process, one port, and no proxy between the browser and the
stream.
