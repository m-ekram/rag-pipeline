# 5. Implementation

## 5.1 Technology stack

The backend is Python 3.11 with FastAPI and Uvicorn. PyMuPDF and pypdf read
PDFs; PaddleOCR (or Tesseract) reads scans; sentence-transformers runs the
embedding and reranking models; SQLite FTS5 provides lexical search; NumPy
holds the dense index. The interface is Next.js 15 with React 19 and
Tailwind CSS. Answer engines are reached over HTTP (Ollama's native API, or
an OpenAI-compatible API for Groq).

## 5.2 Extraction

`PDFExtractor` processes each page in three passes. The first serves pages
from a content-addressed cache or accepts native text whose quality score
(printable, alphabetic and word-count signals, with Devanagari combining
marks counted as letters) passes a threshold; digital pages with tables are
re-read through PyMuPDF's table finder and converted to Markdown. The second
pass OCRs the remaining pages, in parallel worker processes on Linux. The
third stitches tables that continue across page boundaries. OCR failures
fall back to native text but are deliberately not cached, so a transient
failure is retried on the next run; a missing OCR engine is reported as an
error instead of producing blank pages.

## 5.3 Chunking

`StructureAwareParentChildChunker` walks each page:

- **Tables** become children of a few rows each, with the header row and a
  linearised "[Record: column: value …]" form prepended; the parent is the
  whole (possibly stitched) table.
- **Prose** becomes sentence-aligned children of about 200 words, under
  parents capped at 500 words.
- **Electoral pages** are handed to `ElectoralRecordChunker`, which re-aligns
  the column-interleaved OCR output into one record per voter (serial, EPIC,
  name with Latin transliteration, relation, house, age, gender) and groups
  voters by house number as the parent.

## 5.4 Indexes

`FTS5Index` stores chunks in an in-memory SQLite FTS5 table whose tokenizer
keeps `/`, `-`, `_` and `.` inside tokens, so identifiers such as
`BR/35/207/291052` survive. Every query term is quoted, so words such as
NOT or AND cannot be parsed as operators. A field search combines field
markers with groups of dual-script synonyms and falls back to substring
matching for words OCR has glued together.

`LocalDenseIndex` stores normalised float16 vectors and JSON payloads on
disk, written via write-then-rename and checked for consistency on load.
Search is a single matrix-vector product and a partial sort. A fingerprint
of the embedded chunk texts decides whether stored vectors can be reused.

## 5.5 Router and generation

`IntentRouter` classifies each question with ordered patterns (page, ID,
serial, relation, house, administrative, list) and records the intent it
used, which the pipeline reads to size evidence for list questions. Relation
names are cut at the next clause ("… and live in house 4"), and a page
lookup that names a document ("page 66 of the Patna Master Plan", matched
through the initials "pmp") reads only that document's page.

`RAGPipeline` reports each stage to the API, applies the gate, packs the
prompt and streams the answer. `OllamaBackend` sends an explicit context
size that only grows, because changing it reloads the model;
`FallbackBackend` puts a local model behind a hosted one.

## 5.6 API and interface

The API's stream worker runs each request on the pipeline thread, relays its
events through a queue, sends heartbeats while the queue is quiet and marks
the request cancelled when the response stream is closed. A startup hook
loads the embedding and reranking models in the background and reports the
state on `/api/health`.

The interface chooses the engine the server recommends, shows a live stage
and elapsed time while waiting, streams tokens, and shows citations and
per-stage timings under each answer.

## 5.7 Testing

Tests cover the fixed defects and the new components: the local vector index
(ranking, persistence, page filters, crash consistency), heartbeats and
cancellation of the stream worker, context sizing and engine fallback,
OCR-language detection, per-page chunking of a mixed folder, fingerprinted
reuse, and document-aware page lookups.
