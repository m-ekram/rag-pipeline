# Abstract

Retrieval-Augmented Generation (RAG) lets a language model answer questions
from a user's own documents and cite the pages it used. This dissertation
describes *Sanchay*, a local RAG system for three heterogeneous document
families — the 335-page Patna Master Plan 2031, scanned Hindi electoral rolls,
and research papers — and the end-to-end optimisation that made its web
interface usable on a CPU-only laptop.

The system combines native-first PDF extraction with OCR fallback,
structure-aware parent/child chunking (tables, prose sections and per-voter
records), SQLite FTS5 lexical search, multilingual E5 dense retrieval,
intent routing, Reciprocal Rank Fusion, cross-encoder reranking and grounded
generation with validated citations, served by FastAPI to a Next.js
interface.

Tracing a request from browser to model identified nine causes of the
interface's missing and delayed answers, among them a 30-second development
proxy timeout, stream buffering, uncancellable work on a single pipeline
thread, and model loading inside the first request. Fixing them — streamed
heartbeats, cancellation, startup warm-up, a dependency-free persistent vector
index with fingerprinted reuse, per-page chunking, script-aware reranker
selection and engine-aware prompt budgets — was guided by measurements from a
reproducible probe that drives the real HTTP API.

On an Intel i5-8265U laptop, re-indexing an unchanged 335-page document fell
from 317 s to about 7 s, retrieval with reranking from 2.9–3.7 s to 0.8–1.2 s
per question, and the interface now reports progress within a second. With a
local 3-billion-parameter model the median time to the first answer token is
31.6 s (38.5 s to a complete answer), bounded by the CPU's prompt-reading
speed of about 29 tokens per second; a hosted engine is supported for
interactive use. A four-way comparison of prompt configurations showed that
cutting evidence saved little time and cost answers, while one explicit
instruction was needed to stop a small model from replying with citations
alone. Limitations —
chiefly OCR on the test machine and an uncalibrated abstention threshold — are
reported with the evidence behind them.
