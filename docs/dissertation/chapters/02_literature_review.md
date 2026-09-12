# 2. Literature Review

## 2.1 Retrieval-Augmented Generation

Lewis et al. (2020) introduced Retrieval-Augmented Generation, combining a
dense retriever with a sequence-to-sequence generator so that a model's
answers are conditioned on retrieved passages rather than on its parameters
alone. The approach reduces hallucination on knowledge-intensive tasks and,
crucially for this project, makes answers attributable to sources. Later
surveys (Gao et al., 2023) describe the now-standard "advanced RAG" pattern:
pre-retrieval processing, hybrid retrieval, reranking and post-retrieval
prompt construction. Sanchay follows this pattern.

## 2.2 Lexical and dense retrieval

**Lexical retrieval.** BM25 (Robertson and Zaragoza, 2009) ranks passages by
term frequency and inverse document frequency. It remains a strong baseline,
particularly for exact identifiers such as voter EPIC numbers, which dense
models represent poorly. SQLite's FTS5 extension provides BM25 ranking over
an inverted index inside a single file or in memory.

**Dense retrieval.** Dense Passage Retrieval (Karpukhin et al., 2020) and
Sentence-BERT (Reimers and Gurevych, 2019) embed queries and passages into a
shared vector space. The E5 family (Wang et al., 2022) trains such embedders
contrastively on large weakly-supervised pair collections; its multilingual
variant covers Hindi and Urdu, which matters for the electoral rolls. E5
models expect "query:" and "passage:" prefixes — omitting them measurably
reduces recall, a detail the implementation enforces.

**Benchmarks.** BEIR (Thakur et al., 2021) showed that no single retriever
dominates across domains and that BM25 is hard to beat out of domain. FiQA
(Maia et al., 2018), a financial question-answering set within BEIR, is used
by this project's evaluation harness.

## 2.3 Fusion and reranking

Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, 2009) merges rankings
by summing 1/(k + rank) across lists. Because it uses ranks rather than raw
scores, it combines BM25 and cosine similarity, which live on incompatible
scales, without calibration.

Cross-encoders score a query and passage jointly. Nogueira and Cho (2019)
showed that a BERT reranker over BM25 candidates substantially improves
passage ranking on MS MARCO. Distilled models such as MiniLM (Wang et al.,
2020) make this affordable on CPUs. mMARCO (Bonifacio et al., 2021)
translated MS MARCO into several languages, including Hindi, enabling
multilingual cross-encoders. Reranking cost grows with the number of
candidates, model depth and sequence length — the three levers tuned in
Chapter 6.

## 2.4 Document understanding and OCR

Government PDFs mix native text with scanned images. Tesseract (Smith, 2007)
is a long-standing open-source OCR engine with Devanagari support. PaddleOCR
(Du et al., 2020; Cui et al., 2025) provides lightweight detection and
recognition networks, including a Devanagari recogniser, designed for CPU
deployment. Table structure is a separate problem: PyMuPDF exposes table
detection for digital PDFs, and tables spanning pages must be stitched back
together before chunking.

## 2.5 Chunking

How documents are split determines what can be retrieved. Fixed-size windows
are simple but cut tables and records apart. Sentence-aware splitting keeps
sentences whole. Parent-child ("small-to-big") retrieval indexes small units
for precise matching but gives the generator the larger enclosing unit for
context. Sanchay applies this idea per structure: table rows under their
whole table, prose paragraphs under a bounded section, and voter cards under
their household.

## 2.6 Generation, grounding and abstention

Long prompts are not free: Liu et al. (2023) found that models use
information in the middle of long contexts poorly, and on a CPU prompt
evaluation dominates latency. Small instruction-tuned models such as Qwen2.5
(Qwen Team, 2024) make local generation feasible. Grounding is enforced by
asking for numbered citations and validating them against the supplied
evidence.

Selective prediction (Geifman and El-Yaniv, 2017) frames abstention as a
trade between coverage and risk. The project's broader research plan applies
this to RAG under corpus contamination; this dissertation concentrates on the
system and its optimisation, and treats calibrated abstention as future work
(Chapter 7).

## 2.7 Summary

The literature supplies the components: BM25, multilingual dense retrieval,
RRF, cross-encoder reranking, OCR, structure-aware chunking and grounded
prompting. What it does not supply is the engineering needed to make them
work together on heterogeneous documents, on a CPU-only laptop, behind a web
interface that users can trust. That integration and optimisation is the
subject of the following chapters.
