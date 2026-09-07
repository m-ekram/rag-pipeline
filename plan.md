# RAG System — Execution Plan

Companion to `rag_project_revised.docx` (the synopsis). This file is for
day-to-day tracking: what to build, in what order, and what "done" means
at each checkpoint. **Core** = required for a complete, defensible project.
**Stretch** = only if Core finishes early — never let Stretch delay Core.

## Decision log (locked defaults — revisit only if something breaks)

| Choice | Default |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Vector DB | Qdrant, local via Docker |
| Lexical retrieval | `rank_bm25` |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation LLM | Local via Ollama 3.1 (Primary) / Anthropic API (Fallback) |
| Backend | FastAPI + Uvicorn |
| Eval ground truth | BEIR — FiQA subset (finance Q&A, has qrels) |
| Datasets & Messy Data | Governmental Election Lists (Hindi/Devanagari, highly messy) |
| Experiment log | JSON/CSV + pandas, no extra service |

Not deciding these again mid-project is itself a scope-control tool — swap
only on evidence (e.g., BM25 too slow at corpus size → OpenSearch), not vibes.

---

## Phase 0 — Setup (Day 0–1)

- [ ] Repo scaffold: `ingestion/`, `retrieval/`, `rerank/`, `generation/`, `eval/`, `api/`, `tests/`
- [ ] `docker-compose.yml` with Qdrant
- [ ] Pull FiQA subset from BEIR; download/curate the 100–150 doc noisy corpus
- [ ] `.env` for Anthropic API key; confirm a test call works end-to-end
- [ ] pytest + basic CI skeleton (even just a GitHub Actions file that runs `pytest`)

**Checkpoint:** `docker compose up` brings up Qdrant; one script can embed and
upsert a single test document; one script can call the LLM and print a response.

---

## Phase 1 — Retrieval Foundation (Week 1)

### Core
- [ ] PDF/HTML parsing + text cleaning + dedup
- [ ] **Adaptive OCR Integration**: Use PaddleOCR (MobileNet Devanagari, width=1024) to crop and extract text from messy governmental election datasets.
- [ ] **Caching System**: Implement caching for ingested documents to ensure fast re-uploads without re-running expensive OCR.
- [ ] Metadata extraction (doc id, title, section, page) — required for citations later, don't skip
- [ ] Implement 2 chunking strategies: fixed-size and sentence-aware (the other two are Stretch)
- [ ] Embed chunks with bge-small, upsert to Qdrant with metadata as payload
- [ ] BM25 index over the same chunks (`rank_bm25`)
- [ ] Manual smoke test: run 10 hand-picked queries against dense-only and BM25-only, eyeball results

### Stretch
- [ ] Overlapping fixed-size and section-aware chunking
- [ ] Near-duplicate detection beyond exact-match dedup

**Checkpoint:** given a query string, both retrieval paths independently
return ranked chunks with correct source metadata attached.

---

## Phase 2 — Hybrid, Reranking, Generation (Week 2)

### Core
- [ ] RRF fusion combining dense + BM25 rankings
- [ ] Cross-encoder reranking over the fused candidate set
- [ ] **Generation Layer**: Integrate local LLM generation via Ollama (LLaMA 3.1).
- [ ] Grounded generation prompt (evidence-only, no invented facts, cite sources)
- [ ] Citation formatting: `[Document X, Section Y, Page Z]` attached to claims
- [ ] Abstention v1: single threshold on top reranked score (calibrate properly in Phase 3)
- [ ] `POST /query` endpoint wiring the full pipeline together
- [ ] `POST /ingest` endpoint for adding documents

### Stretch
- [ ] Weighted-sum fusion as an alternative to RRF
- [ ] LLM self-check step as a second abstention signal
- [ ] `bge-reranker-base` as a second reranker option, toggleable via config

**Checkpoint:** hitting `/query` with a real question returns an answer,
citations, retrieved evidence, and an abstain/answer decision — and it
correctly abstains on a question the corpus can't cover (e.g., ask about
a topic deliberately absent from the corpus).

---

## Phase 3 — Evaluation & Productionization (Week 3)

### Core
- [ ] Build the labeled eval set: FiQA qrels + a small hand-labeled slice from the noisy corpus (include some no-evidence and irrelevant-evidence queries for abstention testing)
- [ ] Compute Precision@5/10, Recall@5/10, MRR, nDCG@5/10 for BM25, dense, hybrid, hybrid+rerank
- [ ] Run Experiments 1, 2, 4, 5, 6 from the synopsis (Experiment 3 — fusion comparison — is Stretch if time is short)
- [ ] Measure latency per pipeline stage and end-to-end
- [ ] Measure token usage and approximate cost per query
- [ ] Calibrate the abstention threshold against the labeled slice (maximize F1 on abstain/answer)
- [ ] `GET /health`, `GET /metrics`
- [ ] Write up results: tables/plots per experiment, honest limitations section
- [ ] README + API docs

### Stretch
- [ ] Experiment 3 (fusion method comparison)
- [ ] Deploy the API somewhere reachable (Render/Fly/similar) for a live demo

**Checkpoint:** every claim in the final write-up points to a logged
experiment result — no number is asserted without a script that produced it.

---

## Definition of done (project-level)

- [ ] All Core items across Phase 1–3 checked off
- [ ] Metrics table comparing BM25 / dense / hybrid / hybrid+rerank exists and is reproducible by rerunning one script
- [ ] At least 3 abstention cases (sufficient / irrelevant / no evidence) demonstrated and correct
- [ ] **End-to-End Test on Messy Data**: Successfully ingest and retrieve data from the governmental election list PDFs using the full pipeline (OCR + Caching + Ollama).
- [ ] Latency and cost numbers reported, not estimated
- [ ] `rag_project_revised.docx` limitations section matches what was actually observed (update it if reality diverged from the plan)

---

# The goal — what makes this project unique

Everything above describes a competent hybrid RAG system. Competent is not
unique: bge-small + Qdrant + BM25 + RRF + a cross-encoder is the default stack,
and a project that ships it well is still one of ten thousand. The locked
defaults in the decision log stay locked precisely so the *contribution* can
live somewhere other than component selection.

**The goal of this project is to characterise, quantitatively, how a RAG
system's decision to answer or abstain degrades as its corpus fills with
material that does not answer the question — and to show that a properly
calibrated abstention gate converts that degradation from a silent accuracy
collapse into an explicit, predictable loss of coverage.**

Stated as a claim that can be proven wrong:

> An un-gated RAG pipeline's error rate rises sharply with corpus
> contamination, because retrieval always returns *something* and generation
> always uses it. A gated pipeline, with its threshold calibrated on held-out
> data, holds selective risk roughly flat across the same contamination range
> and pays for it in coverage instead. The size of that trade — coverage lost
> per point of error avoided — is the number this project exists to report.

The headline artefact is not a metrics table. It is a **risk–coverage curve per
contamination level**, and the statement "at 90% corpus contamination this
system answers N% of questions and is wrong on M% of what it answers."

## Why this is the unique part

| Most RAG projects | This project |
|---|---|
| Report accuracy on a clean, single-domain corpus | Report accuracy *as a function of corpus contamination* |
| "Handle" not-knowing with a system-prompt instruction | Treat abstention as a calibrated decision with a tuned operating point |
| One abstention outcome: answered / didn't | Three: sufficient evidence / irrelevant evidence / no evidence |
| Single threshold, tuned on the test set | Threshold fit on a held-out calibration split, reported on untouched test |
| Optimise mean accuracy | Optimise *selective* risk — accuracy at a chosen coverage |

## Honest provenance — what is borrowed and what is ours

Do not oversell this in the write-up. Risk–coverage analysis and selective
prediction are established (Geifman & El-Yaniv and successors); so is the
observation that retrievers degrade with distractors. Neither is invented here.

What is ours is the **combination and the measurement discipline**: applying
selective-prediction methodology to a RAG pipeline under a *controlled,
swept* contamination variable, with every reported number produced by a
rerunnable script. The novelty claim is "nobody bothers to measure this
carefully in an applied RAG build", not "this metric is new".

## The unique deliverable: the contamination × coverage grid

The corpus we already have makes this cheap. FiQA ships 57,638 documents of
which only 1,706 are judged by the test qrels, across 648 test queries
(~2.6 judged docs per query). That leaves **55,932 unjudged in-domain
documents** — a free, topically plausible hard-negative pool.

Two distractor types, deliberately distinguished:

- **In-domain distractors** — unjudged FiQA docs. Same vocabulary, same domain,
  genuinely hard. This is where reranking should earn its cost.
- **Out-of-domain distractors** — the noisy corpus. Topically alien, easy to
  reject. This is where abstention should earn its cost.

Sweep, holding the 648 queries and their 1,706 judged docs fixed in every run:

| Level | Added in-domain distractors | Contamination ρ |
|---|---|---|
| ρ0 | 0 (judged docs only) | 0.00 |
| ρ1 | 1,706 | 0.50 |
| ρ2 | 15,354 | 0.90 |
| ρ3 | 55,932 (full corpus) | 0.97 |

Run the full BM25 / dense / hybrid / hybrid+rerank ladder at every level, gated
and un-gated. That is one nested loop over an eval script that Phase 3 already
requires — the sweep is the contribution, and it is nearly free.

**Open decision, and it bites here:** the out-of-domain axis needs roughly
1,500–15,000 documents to reach contamination ratios comparable to the table
above. 100–150 self-collected PDFs caps ρ_ood at about 0.08, which is too small
to show anything. Either accept that the OOD axis is qualitative only (a
demonstration, not a curve), or source the OOD corpus at a scale the sweep can
use. This is the same PDF-vs-scraped-text fork as Phase 1 ingestion; decide it
once, for both reasons.

## Metrics this adds

| Metric | Definition | Why |
|---|---|---|
| Coverage `c` | fraction of queries answered rather than abstained | the axis everything else is read against |
| Selective risk `R(c)` | error rate *among answered queries* | the number that actually matters to a user |
| `R@c` for c ∈ {1.0, 0.8, 0.6, 0.4} | risk at fixed coverage | comparable across pipelines and ρ levels |
| AURC | area under the risk–coverage curve | one scalar per configuration; lower is better |
| Coverage cost | coverage lost per point of risk avoided | the explicit statement of the trade |
| ΔAURC vs ρ | how AURC moves as contamination rises | **the thesis, in one plot** |

Keep the existing P@k / R@k / MRR / nDCG@k table — it measures the retriever.
These measure the *system's judgement*, which is the different thing.

## Calibration discipline (non-negotiable)

The entire claim collapses if the threshold is fitted on the data it is
reported on. Therefore:

- [ ] Split queries into **train / calibration / test** before any tuning
- [ ] Fit every abstention threshold on **calibration only**
- [ ] Report on **test only**, and report it once
- [ ] Recalibrate per ρ level, and additionally report the ρ0-fitted threshold
      applied unchanged at ρ3 — the realistic deployment case, where the corpus
      degrades after you shipped
- [ ] Log threshold, split seed, and corpus manifest hash with every result row

That fifth item is what makes the grid reproducible rather than anecdotal.

## Definition of unique-done

- [ ] Risk–coverage curve plotted for all four pipeline variants at all four ρ levels
- [ ] AURC reported per (pipeline, ρ) cell, in one table, from one script
- [ ] The coverage cost of holding risk at a fixed target stated as a single sentence with a number in it
- [ ] Un-gated vs gated compared at identical ρ, showing where the un-gated curve breaks
- [ ] The ρ0-calibrated-threshold-applied-at-ρ3 result reported honestly, including if it is bad
- [ ] Three abstention classes (sufficient / irrelevant / no evidence) each demonstrated with a real query and its retrieved evidence
- [ ] Write-up states plainly which parts are borrowed methodology and which are this project's measurement

## What would falsify the thesis

Record these outcomes if they happen; a negative result reported clearly is
worth more than a positive result reported loosely.

- The gated and un-gated risk curves stay within noise of each other as ρ rises
  → the gate is not doing useful work, and the top-reranked-score signal is too
  weak to calibrate on.
- AURC is flat across ρ0→ρ3 → contamination does not degrade this pipeline at
  this scale, and the sweep's premise is wrong.
- Recalibrating per ρ level barely beats the fixed ρ0 threshold → calibration is
  unnecessary, which is itself a useful, publishable-in-a-README finding.
- Abstention triggers almost entirely on out-of-domain distractors and never on
  in-domain hard negatives → the system detects topic drift, not insufficiency,
  and the three-class taxonomy is not actually separable by this signal.

## Scope note

This goal adds **one** genuinely new piece of work: the contamination sweep loop
and the risk–coverage plotting. Everything else is a reframing of Phase 3 Core
items that already exist — the abstention threshold moves from "single F1-tuned
number" to "curve", and the eval script gains an outer loop over corpus
composition. It does not touch Phases 1–2. If Phase 3 is running late, cut ρ1
and ρ2 and report ρ0 vs ρ3 only; the claim survives with two points, weakly.

**Never let this goal delay Core Phase 1–2 work.** A calibrated abstention gate
with nothing behind it is not a project.
