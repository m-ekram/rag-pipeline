# 4. Methodology

## 4.1 Approach

The work followed an engineering methodology of *measure, diagnose, change,
re-measure*. Each change was motivated by a specific observed failure or
measured cost, implemented with a regression test that fails on the previous
code, and re-measured on the same hardware. Where a measurement could not be
taken — for example the original web interface, which could not run on the
test machine at all — this is stated rather than estimated.

## 4.2 Test environment

| Item | Value |
|---|---|
| CPU | Intel Core i5-8265U, 4 cores / 8 threads, 1.6 GHz base |
| Memory | 16 GB |
| GPU | None used (MX130 not suitable) |
| OS | Windows 11 Pro |
| Python | 3.11.9; PyTorch 2.10 (CPU) |
| Embedding model | intfloat/multilingual-e5-small |
| Rerankers | cross-encoder/ms-marco-MiniLM-L-6-v2; cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 |
| Answer engine | Ollama 0.34 with qwen2.5:3b (Q4_K_M, 1.9 GB) |
| OCR | PaddleOCR 3.7 with PP-OCRv5 mobile detection and Devanagari recognition |

## 4.3 Corpora

A demonstration corpus was assembled with one representative of each family:

| Folder | Document | Pages | Type |
|---|---|---|---|
| `data/demo/pmp` | Patna Master Plan 2031 report | 335 | Digital, table-heavy |
| `data/demo/rolls` | Electoral roll, AC 183, part 1 | 28 | Scanned Hindi |
| `data/demo/paper` | Research paper | 18 | Digital prose |

## 4.4 Questions

Five questions per folder (`eval/golden_questions.json`) were chosen to
exercise different retrieval routes: table lookups, page lookups and
open-ended questions for the Master Plan; polling-station, count, house and
serial lookups for the roll; contribution, method, findings, limitations and
future work for the paper.

## 4.5 Measurement protocol

Latency is measured by `eval/latency_probe.py`, which drives the running API
exactly as the browser does. For indexing it records the total time and the
server's progress log. For every question it records, on the client side,
the time to the first byte, the first status line, the first answer token
and the complete answer; and on the server side the retrieval, reranking and
generation times reported in the final event. Each run writes a JSON and CSV
file to `docs/dissertation/data/`; every figure in Chapter 6 is drawn from
those files by `docs/dissertation/make_figures.py`.

Component costs that the API does not expose (embedding throughput, reranker
configurations, OCR per page) were measured with small standalone
benchmarks, run one at a time to avoid contention.

## 4.6 Correctness

The automated test suite (289 tests after this work) runs without models or
servers, using fakes for the embedder, vector store and answer engine. Every
bug fixed during the project has a test that fails against the previous code;
this was verified by running the new tests against the unmodified code.
