# Hindi electoral-roll retrieval — benchmark results

**Date:** 2026-09-04 · **Corpus:** Bihar SIR Final Roll 2025, AC-183 कुम्हरार
**Scripts:** [`eval/electoral_pipeline.py`](../../eval/electoral_pipeline.py),
[`eval/compare_bm25_dense.py`](../../eval/compare_bm25_dense.py)

Every number below was produced by one of those two scripts on the machine
described in *Environment*. Nothing here is estimated.

---

## Headline findings

1. **BM25 abstains cleanly on this corpus; dense retrieval never abstains.**
   BM25 returns nothing for 6/6 nonsense queries. Dense returns confident-looking
   results for all of them, separable only by a **+0.019** margin.
2. **Three separate bugs silently deleted Devanagari** between OCR and retrieval.
   OCR was never the problem — everything downstream of it was.
3. **Caching turns a 412s ingest into 0.0s**, which is what makes iterating on
   chunking and retrieval practical at all.
4. **Dense retrieval underperforms on this corpus** despite working correctly on
   isolated phrases. Cross-lingual is its one clear win.

---

## Environment

| | |
|---|---|
| Machine | Apple M4, 16 GB unified memory, 10 cores, MPS available |
| Python | 3.12.9 |
| OCR | PaddleOCR — `PP-OCRv5_mobile_det` + `devanagari_PP-OCRv5_mobile_rec` |
| OCR recognition width | `max_imgW = 1024` |
| Page render | PyMuPDF at 1.5× → 893 × 1263 px |
| Embeddings (multilingual) | `intfloat/multilingual-e5-small`, 384-dim |
| Embeddings (baseline) | `BAAI/bge-small-en-v1.5`, 384-dim |
| Vector store | Qdrant, in-memory |

### Data

| File | Pages | Native text |
|---|---|---|
| `...HIN-1-WI.pdf` | 28 | 0 chars |
| `...HIN-4-WI.pdf` | 38 | 0 chars |
| `...HIN-5-WI.pdf` | 34 | 0 chars |
| `...HIN-6-WI.pdf` | 44 | 0 chars |
| **Total** | **144** | — |

Every page is a pure scan: `pypdf` extracts **zero characters** from all of them,
so 100% of pages route through OCR. This is what makes caching load-bearing
rather than a convenience.

---

## OCR extraction

Measured on `HIN-1-WI.pdf` (28 pages).

| Metric | Value |
|---|---|
| Total extraction time (cold) | **412.4 s** |
| Per page | **14.73 s** |
| Pages routed to OCR | 28 / 28 |
| Mean OCR confidence | **0.9039** |
| Single-page timing (page 1) | 6.2 s, confidence 0.9516, 1354 chars |
| PaddleOCR init | 4.9 s |

Sample page-1 output (verbatim):

> निर्वाचक नामावली 2025 S04 बिहार
> विधानसभा क्षत्र की संख्या, नाम और आरक्षण की स्थिति : 183 - कुम्हरार (सामान्य)

Recognition is accurate on headers and structured fields. Quality degrades on
dense voter tables — visible as dropped matras (`क्षत्र` for `क्षेत्र`,
`नि्वाचक` for `निर्वाचक`) and mis-read digits. Sufficient for retrieval;
**not** sufficient for extracting individual voter records verbatim.

## Extraction cache

| Run | Time | Cache |
|---|---|---|
| First (cold) | 412.4 s | 0 hits / 28 misses |
| Second | **0.0 s** | **28 hits / 0 misses (100%)** |

Cache size: 296 KB for 28 pages. Keyed on file **content hash** + page + every
setting that changes output (render scale, OCR language, thresholds, provider),
so a hit guarantees byte-identical text to a fresh run. A renamed file still
hits; an edited file correctly misses.

---

## Bugs found

All three were silent — no exception, no warning, no empty result.

### 1. BM25 tokenizer discarded all Devanagari

`retrieval/bm25.py` used `[a-z0-9]+`:

```
tokenize("निर्वाचक नामावली 2025 बिहार")  →  ['2025']
```

The first fix — plain `\w+` — was **also wrong**: Python's `\w` follows
`str.isalnum()`, which is `False` for combining marks (categories `Mn`/`Mc`), so
words split at every matra:

```
→ ['न', 'र', 'व', 'चक', 'न', 'म', 'वल', '2025', ...]
```

Fixed with a character class covering 1,015 combining marks:

```
→ ['निर्वाचक', 'नामावली', '2025', 'बिहार', 'कुम्हरार', 'पटना', 'साहिब']
```

### 2. Clean Hindi was scored as corrupted

Matras are not `isalpha()`, so `_suspicious_ratio` treated them as garbage
characters:

| Text | `suspicious_ratio` | `alpha_ratio` | `needs_ocr` |
|---|---|---|---|
| Hindi (before fix) | 0.3167 | 0.4833 | `True` |
| English (same meaning) | 0.0149 | ~0.83 | `False` |
| Hindi (after fix) | **0.0000** | **0.7872** | **`False`** |

`needs_ocr` fired purely from `suspicious_ratio > 0.15`, forcing OCR on text that
was already perfect. Harmless on this scanned corpus, but it would have wrecked
any born-digital Hindi PDF.

### 3. English-only stoplist broke BM25 abstention

The Hindi particle `का` acted as a content term. Two unrelated nonsense queries
scored **identically at 2.295** — above the genuine query `कुम्हरार` at 1.560:

| Query | Before | After |
|---|---|---|
| `बिल्ली का बच्चा कहाँ सोता है` (nonsense) | 2.295 | **no results** |
| `मेरी कार का इंजन खराब है` (nonsense) | 2.295 | **no results** |
| `कुम्हरार` (genuine) | 1.560 | 1.560 |

80 Hindi function words added; domain vocabulary (मतदान, निर्वाचक, नाम, संख्या,
क्षेत्र) deliberately excluded, enforced by a test. False matches: **2/6 → 0/6**.

### 4. Embedding model cannot represent Hindi

`bge-small-en-v1.5` emits **no `[UNK]` tokens** on Devanagari — which is why the
failure is silent — but shatters words into characters
(`निर्वाचक` → `न ##ि ##र ##व ...`). Cosine similarity, same word pair:

| Model | related | unrelated | separation |
|---|---|---|---|
| `bge-small-en-v1.5` | 0.4079 | 0.3747 | **0.033** (noise) |
| `multilingual-e5-small` | 0.8826 | 0.7867 | **0.096** |

`multilingual-e5-small` is also 384-dim, so the Qdrant vector size is unchanged.

---

## BM25 vs dense

123 chunks (fixed-size, 180 words, 40 overlap) from `HIN-1-WI.pdf`.
Dense index build: 19.3 s.

### Top-1 scores

| | Query | BM25 | Dense |
|---|---|---|---|
| ✅ | मतदान केंद्र | **15.589** | 0.865 |
| ✅ | पटना साहिब | **10.450** | 0.848 |
| ✅ | निर्वाचक नामावली | **9.617** | 0.877 |
| ✅ | मतदाता का नाम | 5.693 | 0.871 |
| ✅ | विधानसभा क्षेत्र की संख्या | 4.637 | 0.881 |
| ✅ | भाग संख्या | 3.947 | 0.840 |
| ✅ | कुम्हरार | 1.560 | 0.817 |
| ❌ | How do I bake sourdough bread? | **—** | 0.743 |
| ❌ | रोटी कैसे बनाएं | **—** | 0.798 |
| ❌ | quantum chromodynamics lagrangian | **—** | 0.743 |
| ❌ | बिल्ली का बच्चा कहाँ सोता है | **—** | 0.789 |
| ❌ | best football team in brazil | **—** | 0.727 |
| ❌ | मेरी कार का इंजन खराब है | **—** | 0.770 |

### Abstention separability — the key result

| | BM25 | Dense |
|---|---|---|
| Answerable, min score | 1.560 | 0.817 |
| Nonsense, max score | *none returned* | 0.798 |
| Nonsense queries returning results | **0 / 6** | **6 / 6** |
| Separation | **categorical** | **+0.019** |

BM25's abstention needs no threshold: no matching term, no result. Dense requires
a threshold calibrated inside a 2%-wide window, on a score range compressed into
0.727–0.881.

**For a project whose thesis is knowing when *not* to answer, this is the
significant result — and it favours the lexical path.**

### By query type

| Group | BM25 returns | Dense returns | top-k overlap | BM25 | Dense |
|---|---|---|---|---|---|
| lexical | 3/3 | 3/3 | 0.39 | 0.1 ms | 23.2 ms |
| paraphrase | 1/3 | 3/3 | 0.00 | 0.1 ms | 17.2 ms |
| crosslingual | 2/3 | 3/3 | 0.07 | 0.1 ms | 9.5 ms |
| unanswerable | **0/2** ✅ | 2/2 ❌ | 0.00 | 0.1 ms | 12.4 ms |

BM25 is ~100–200× faster per query.

### Where dense wins, and where it fails

**Wins — cross-lingual.** `"electoral roll 2025 Bihar"` → correct page 1 at
0.892. `"list of voters and their names"` returns results where BM25 returns
nothing. BM25 cannot bridge scripts at all.

**Fails — Hindi paraphrase.** `"वोटिंग बूथ कहाँ है"` returned voter-name tables,
**not** the polling-station page that BM25 finds instantly under the literal term
`मतदान केंद्र`. Likely cause: these pages are dense tables of names and numbers,
so most chunks look semantically alike to the embedder, and OCR noise degrades
semantics more than it degrades keyword matching.

---

## Limitations

- **No qrels for this corpus.** Everything above is a capability diagnostic, not
  a metrics table. Precision/Recall/nDCG require hand-labelled relevance
  judgements that do not yet exist for the electoral rolls.
- **Small query set.** 13 queries, hand-written by the authors. The +0.019 dense
  margin in particular is measured on 13 points and should not be treated as a
  calibrated threshold.
- **Single PDF.** Comparison ran on `HIN-1-WI.pdf` (28 pages, 123 chunks) only.
- **OCR quality is not measured against ground truth.** Confidence 0.9039 is
  PaddleOCR's *self-reported* score, not accuracy against a transcript.
- **`max_imgW = 1024`** is inherited from an earlier benchmark whose script is
  not in the repository. It is currently an unreproducible constant.

## Full-corpus results (144 pages, 600 chunks)

All four PDFs, loaded from cache in **0.1 s**. The single-PDF findings hold:

| | BM25 | Dense |
|---|---|---|
| Answerable, min score | 1.587 | 0.825 |
| Nonsense returning results | **0 / 6** | **6 / 6** |
| Separation | categorical | **+0.0272** |

### Record-aware chunking

Replacing 180-word windows with row-band chunks that repeat the page header
(`ingestion/electoral_chunking.py`), on the same 144 pages:

| | fixed-180/40 | electoral-2band |
|---|---|---|
| Chunks | 600 | 1,823 |
| Mean chunk size | 163.4 w | 62.6 w |
| Chunks carrying constituency context | **0 / 600** | **1,823 / 1,823** |
| EPIC numbers tagged as metadata | 0 | 3,424 |
| Dense abstention margin | +0.0272 | **+0.0523** |
| BM25 nonsense returned | 0 / 6 | 0 / 6 |
| Exact EPIC lookup `SHS4252763` | 5.76 | **6.32** |
| Voter name `मधु गुप्ता` | 7.17 | **9.37** |
| Section `गोविन्द मित्रा रोड` | 8.47 | **2.21** |

**The dense separation margin nearly doubles.** Missing page context was the
reason chunks looked alike to the embedder — the original hypothesis, confirmed.

The section-name regression is real and expected: repeating the header in every
chunk collapses those terms' IDF, so they stop discriminating between chunks.
`include_header=False` exists for callers who prefer the opposite trade.

### Cross-lingual dense rescue is not fully viable

BM25 cannot bridge scripts, so a dense fallback was tested for queries where
BM25 is silent. The ranges overlap:

| Cross-lingual (should admit) | | Nonsense (must exclude) | |
|---|---|---|---|
| Where is the polling station? | **0.7815** | रोटी कैसे बनाएं | **0.7923** |
| voter identity card number | 0.8279 | मेरी कार का इंजन खराब है | 0.7658 |
| list of voters and their names | 0.8383 | बिल्ली का बच्चा कहाँ सोता है | 0.7646 |
| electoral roll 2025 Bihar | 0.8817 | best football team in brazil | 0.7083 |

`"Where is the polling station?"` scores **below** the highest nonsense query, so
**no floor admits all cross-lingual queries while excluding all nonsense**. A
floor at 0.80 is the best trade available: 3 of 4 cross-lingual recovered, 0 of 6
nonsense admitted. Rescue is therefore off by default.

### What the results changed in the code

- `retrieval/cascade.py` — `LexicalFirstRetriever`. BM25 runs first; dense fuses
  via RRF **only when BM25 already found something**. Unconditional fusion would
  destroy abstention, because dense returns results for every query.
- `ingestion/electoral_chunking.py` — `ElectoralRecordChunker`. Splits on
  serial/EPIC row-band boundaries and repeats the page header.

Per-voter chunking is deliberately not attempted: OCR flattens the three-column
table column-major (three serials, then three EPICs, then three names), and
fields are dropped often enough that positional re-association would silently
attach the wrong father's name to a voter.

## Reproducing

```bash
python eval/electoral_pipeline.py            # OCR -> chunk -> BM25, cached
python eval/compare_bm25_dense.py            # BM25 vs dense comparison
python eval/compare_bm25_dense.py --all-pdfs  # full 144-page corpus
```

First run costs ~35 min of OCR for the full corpus; every later run is served
from `.cache/extraction/`.
