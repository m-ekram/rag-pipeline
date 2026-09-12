# 1. Introduction

## 1.1 Background

Public administration in India produces documents that are rich in facts but
hard to query. A city's master plan runs to hundreds of pages of tables and
maps; an electoral roll lists every voter of a polling station as scanned
Devanagari cards; research on urban policy is published as dense academic
prose. A citizen, journalist or official who needs one fact from these
sources — the proposed share of residential land, the voters registered in a
house, a study's main finding — must today read the document by hand.

Retrieval-Augmented Generation (RAG) promises a better interface: a language
model answers questions in natural language, but only from passages retrieved
from the user's own documents, and cites them. The promise is attractive for
government records precisely because the answer must be traceable to a page.
It is also demanding, because the documents are heterogeneous (digital text,
tables, scans), multilingual (English, Hindi, Urdu), and must often be
processed on modest hardware without sending them to a cloud service.

## 1.2 Problem statement

This project, *Sanchay* (संचय, "collection"), began as a hybrid RAG pipeline
with a command-line interface and a Next.js web interface. The pipeline
produced answers on the terminal, but the web interface either showed nothing
or answered only after several minutes. The system had to be made to work
end to end, from browser to model and back, for three document families at
once:

- the **Patna Master Plan 2031**, a 335-page digital report with many tables;
- the **Bihar electoral rolls** of assembly constituency 183, 435 scanned
  Hindi PDFs of about 28 pages each;
- **research papers**, digital prose documents.

## 1.3 Aims and objectives

The aim is a local RAG system that answers questions over any of these
corpora quickly, visibly and with verifiable citations. The objectives are:

1. Analyse the architecture end to end and identify every cause of missing or
   slow answers in the web interface.
2. Remove those causes and optimise each pipeline stage — extraction, OCR,
   chunking, indexing, retrieval, reranking and generation — for a CPU-only
   laptop.
3. Make retrieval behave correctly for each document type: tables, per-voter
   records and prose, including folders that mix them.
4. Measure latency per stage with a reproducible script, so that every number
   reported here can be regenerated.
5. Document the architecture and the engineering decisions.

## 1.4 Scope and constraints

All work was carried out on an Intel Core i5-8265U laptop (4 cores, 8
threads, 16 GB RAM, no usable GPU) running Windows 11. Several constraints
of that environment shaped the design and are reported honestly: Windows
Application Control blocked a compiled dependency of the Qdrant client; the
installed PaddlePaddle release failed in its accelerated CPU path; and the
embedding and language models run on the CPU only. The answer engine is a
local Ollama model (qwen2.5:3b); a hosted engine (Groq) is supported but was
not configured for the measurements.

## 1.5 Contributions

- A diagnosis of why a streaming RAG interface fails behind a development
  proxy, and a design that streams progress, heartbeats and tokens reliably
  and cancels abandoned work.
- A dependency-free, persistent dense index that replaces an external vector
  database on a single machine, with content-fingerprinted reuse.
- Per-page, structure-aware chunking that keeps tables, prose sections and
  voter records intact in a single folder.
- Latency optimisations — model reuse, context sizing, script-aware prompt
  budgets, reranker selection by script share — each justified by a
  measurement.
- A reproducible latency probe that drives the real HTTP API.

## 1.6 Structure of this report

Chapter 2 reviews the relevant literature. Chapter 3 describes the system
architecture. Chapter 4 sets out the methodology: the corpora, questions and
measurement protocol. Chapter 5 covers the implementation. Chapter 6
presents the optimisations and the measured results. Chapter 7 discusses
limitations and future work, and Chapter 8 concludes.
