# 6. Optimisation and Results

This chapter follows the pipeline from the browser inwards. Each section
states what was observed, what was changed and what the change measured.
Raw data for every number is in `docs/dissertation/data/`.

## 6.1 Why the web interface showed nothing

The command-line tool printed answers, so the pipeline itself worked; the
failures were in how the interface reached it. Tracing one request end to end
exposed nine causes:

| # | Where | Cause | Effect on the user |
|---|---|---|---|
| 1 | Next.js dev proxy | 30-second proxy timeout; gzip compression of the stream | Long requests cut off; progress and tokens held back until the end |
| 2 | API worker | One pipeline thread and no cancellation | An abandoned answer kept generating; later questions queued silently |
| 3 | First request | Library imports, model loads and a ~1 GB model download inside the first "Index" | Minutes of silence on first use |
| 4 | Vector store | Qdrant client blocked by Application Control; embedded Qdrant locked per index | Indexing failed on the laptop; re-indexing failed elsewhere |
| 5 | Answer engine | No context size sent to Ollama; interface picked an arbitrary model | Long prompts silently truncated; slow model by default |
| 6 | API indexing | One OCR worker, English OCR for Hindi scans, top-level files only | Hindi rolls read as noise; nested data ignored |
| 7 | Chunking | One electoral page switched the whole folder to the voter-card chunker | Master Plan tables lost in mixed folders |
| 8 | Page lookups | Page filters ignored the document | "Page 66" mixed pages from every file |
| 9 | Feedback | No events while the pipeline was busy | Slow steps looked like a crash |

The fixes (Chapter 5) stream a heartbeat every two seconds with the current
stage, cancel work when the browser disconnects, warm the models at startup,
and — in production — serve the interface from the API process itself, so no
proxy sits between the browser and the stream. Automated tests check the
heartbeat and cancellation behaviour, and the production build was verified
to be served by FastAPI at the API's own address.

## 6.2 Indexing

![Where first-index time goes](../figures/fig_index_time.png)

| Folder | Pages | Chunks | First index | Of which embedding | Re-index (unchanged) |
|---|---|---|---|---|---|
| Research paper | 18 | 83 | 59 s | 16 s | 45–53 s (see note) |
| Master Plan | 335 | 1,616 | 317 s | 316 s | 6.6–7.0 s |

Note: the paper's re-index time is dominated by loading the answer model into
memory before the first question, not by indexing itself.

Embedding dominates the first index: the multilingual E5 model embedded
8–10 chunks per second on this CPU in a standalone benchmark, regardless of
the thread count. Extraction of the digital Master Plan was already served
from the page cache (1–2 s for 335 pages). Fingerprinting the chunk texts
lets a re-opened folder skip embedding entirely: the Master Plan re-index fell
from 317 s to about 7 s.

## 6.3 Retrieval and reranking

With a stub engine (no generation), the time from question to evidence was
1.3–1.8 s for the research paper and 2.9–3.7 s for the Master Plan; a page
lookup took 0.03 s because it bypasses ranking entirely. The Master Plan was
slower because a few quoted Hindi words had selected the 12-layer
multilingual reranker.

![Cross-encoder reranking cost on CPU](../figures/fig_rerankers.png)

| Reranker | 384 tokens | 256 tokens |
|---|---|---|
| ms-marco-MiniLM-L-6 (English) | 1.04 s | 1.06 s |
| mmarco-mMiniLMv2-L12 (multilingual) | 4.32 s | 2.21 s |

Two changes followed: the multilingual model is chosen only when Devanagari
or Arabic letters make up at least 15% of the text, and reranker input is
capped at 256 tokens. With the local model, retrieval for Master Plan
questions then took 0.8–1.2 s (median about 1.0 s across all questions).

## 6.4 Generation

Generation was measured directly against Ollama with real Master Plan
evidence:

| Model | Prompt reading | Output | Repeat of an identical prompt |
|---|---|---|---|
| qwen2.5:3b | 24–30 tokens/s | 6.4–6.7 tokens/s | 0.17 s (prompt cache) |
| qwen2.5:1.5b | 51 tokens/s | 13–13.5 tokens/s | 0.08 s |

On this CPU a 3-billion-parameter model reads about 29 prompt tokens per
second, so a 1,000-token prompt waits roughly half a minute for its first
token. The smaller model is 1.7× faster but answered the residential
land-use question wrongly (40% instead of 55.04%), so it was rejected as the
default. Ollama reuses the cached system prompt between questions, which
makes the length of the evidence — not of the instructions — the
per-question cost.

Four prompt configurations were run through the full API with the ten golden
questions:

![Prompt configuration vs latency and answers](../figures/fig_run_comparison.png)

| Run | Configuration | Median prompt | Median first token | Median total | Answered | Answers stating the fact |
|---|---|---|---|---|---|---|
| A | Original ten-rule prompt, 1,200-token evidence with parent sections | 1,230 tok | 34.4 s | 40.6 s | 8/10 | 8/8 |
| B | Compact prompt, 700-token evidence, no parent sections | 864 tok | 31.9 s | 37.3 s | 7/10 | 6/7 |
| C | Compact prompt, 1,200-token evidence with parent sections | 1,006 tok | 27.2 s | 31.0 s | 8/10 | 6/8 |
| D | C with an "answer in sentences" rule (final) | 1,027 tok | 31.6 s | 38.5 s | 8/10 | 8/8 |

Three findings came out of the comparison:

- **Cutting evidence was a poor trade.** Configuration B shortened prompts by
  30% but the first token by only 7%, and a Master Plan question lost its
  answer. Parent sections carry the facts a small model needs.
- **Shorter instructions changed behaviour, not only speed.** With the
  compact prompt the 3B model twice replied with a bare citation — "[Document
  pmp-2031-report, Page 80]" for the residential share, "[5]" for the road
  widths — which the answer counter scores as "answered" but which states no
  fact. C's lower median also includes one exact repeat of a B prompt served
  from Ollama's cache (1.0 s).
- **One explicit rule fixed it.** Asking for sentences that state the fact,
  with an example ("Residential use is 55.04% [1]"), restored every factual
  answer while keeping the shorter prompt. D is the final configuration.

![Answer latency per question with the final configuration](../figures/fig_answer_latency.png)

## 6.5 OCR

The scanned electoral rolls could not be measured on the test laptop. The
installed PaddlePaddle 3.3.1 fails in its oneDNN path; without oneDNN a roll
page took 104–118 s with good recognition (216 lines, 0.93 mean confidence).
PaddlePaddle 3.0.0 was tried in isolation and conflicted with PyTorch's
runtime. Tesseract, the pipeline's fallback engine, required a machine-wide
installation that was not completed. The pipeline now disables a failing
Paddle engine after its first error and reports a missing OCR engine as an
error instead of producing blank pages — which is exactly how the original
interface had been failing silently on the rolls.

## 6.6 Answer quality

With the final configuration the system answered all five research-paper
questions with cited facts (for example, the study's annotated dataset of
1,200 images from several clinics, used to train a YOLOv3 model at 99.33%
accuracy) and three of five Master Plan questions: the proposed residential
land-use share of 55.04% (page 80), a projected 2031 population of 60.25 lakh
for the planning area (page 78), and the widths of the proposed road
hierarchy (80 m, 45 m and 30.5 m). The page-66 question and the question
about implementing agencies were declined in every run: the evidence was
retrieved, but the 3B model judged it insufficient. Declining is the designed
behaviour, and the interface presents it as an abstention rather than an
error.

## 6.7 Summary

| Stage | Before | After |
|---|---|---|
| Interface feedback | none until the request ended (or was cut off at 30 s) | status within 1 s, heartbeat every 2 s |
| First use | imports, model loads and downloads inside the first request | models warmed at startup; prefetch script |
| Re-index of an unchanged folder | full re-embedding (317 s for the Master Plan) | ~7 s |
| Retrieval + rerank (Master Plan) | 2.9–3.7 s | 0.8–1.2 s |
| Answer, local 3B model (median) | first token 34.4 s, complete 40.6 s | first token 31.6 s, complete 38.5 s, all answers factual |

On this laptop the answer model is now the only slow stage, and its cost is
set by the CPU's prompt-reading speed of about 29 tokens per second. For
interactive use the hosted engine (Groq), which the system selects
automatically when a key is configured, removes that bottleneck; the local
model remains the offline fallback.
