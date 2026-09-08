"""Interactive CLI tool to ingest any document and query it via the RAG pipeline.

Supports:
- PDFs (native text extraction + automatic PaddleOCR fallback with caching)
- Plain text (.txt) and Markdown (.md) files
- Hybrid Lexical-First retrieval (BM25 + Qdrant dense embeddings)
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- Calibrated Abstention gating (answers only when evidence is sufficient)
- Grounded generation with source citations via Ollama (llama3.1:8b)

Usage:
    # Interactive Q&A:
    python ask.py path/to/document.pdf

    # Single question:
    python ask.py path/to/document.pdf -q "What is the key takeaway?"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from generation.abstention import Decision, ThresholdGate
from generation.citations import render_citations
from console import use_utf8_console
from generation.llm import GroqBackend, OllamaBackend, get_llm
from generation.pipeline import RAGPipeline
from ingestion.chunking import (
    ElectoralRecordChunker,
    FixedSizeChunker,
    SentenceAwareChunker,
    StructureAwareParentChildChunker,
)
from ingestion.documents import Chunk, Document
from ingestion.electoral import is_electoral_text
from ingestion.pdf_extractor import PDFExtractor
from ingestion.pipeline import chunk_corpus
from rerank.cross_encoder import (
    CrossEncoderReranker,
    DEFAULT_MODEL,
    MULTILINGUAL_BASE,
    MULTILINGUAL_LIGHT,
)
from retrieval.bm25 import BM25Index
from retrieval.cascade import LexicalFirstRetriever
from retrieval.dense import DenseIndex
from retrieval.embedder import Embedder
from retrieval.fts5_index import FTS5Index
from retrieval.router import IntentRouter


def load_file(file_path: Path, ocr_lang: str = "en", max_pages: Optional[int] = None, workers: int = 1) -> list[Document]:
    """Load a PDF, TXT, or MD file into Document instances."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        page_info = f" (max_pages={max_pages})" if max_pages else ""
        print(f"[*] Extracting text from PDF: {file_path.name}{page_info} (OCR lang: {ocr_lang}, workers: {workers})...")
        start = time.perf_counter()
        extractor = PDFExtractor(ocr_lang=ocr_lang, use_cache=True, workers=workers)
        docs = list(extractor.extract(str(file_path), clean=True, max_pages=max_pages))
        elapsed = time.perf_counter() - start
        print(f"[+] Extracted {len(docs)} pages in {elapsed:.2f}s.")
        return docs

    elif suffix in (".txt", ".md", ".csv", ".json", ".log"):
        print(f"[*] Reading text file: {file_path.name}...")
        text = file_path.read_text(encoding="utf-8", errors="replace")
        doc_id = file_path.stem
        return [
            Document(
                doc_id=doc_id,
                text=text,
                metadata={"filename": file_path.name, "path": str(file_path)},
            )
        ]
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .pdf, .txt, .md")


def sanitize_collection_name(file_path: Path) -> str:
    """Ensure collection name meets Qdrant conventions and is content-addressed."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", file_path.stem)
    try:
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:8]
    except Exception:
        digest = "default"
    return f"doc_v4_{clean[:17]}_{digest}"


def build_pipeline(
    docs: list[Document],
    collection_name: str,
    *,
    chunk_size: int = 200,
    overlap: int = 40,
    threshold: float = -2.0,
    use_reranker: bool = True,
    reranker_model: Optional[str] = None,
    backend: Optional[str] = "auto",
    model: Optional[str] = None,
    reindex: bool = False,
    qdrant_client=None,
) -> RAGPipeline:
    """Build chunks, indexes, retriever, and RAGPipeline."""
    has_electoral = any(is_electoral_text(d.text) for d in docs)
    if has_electoral:
        print("[*] Detected electoral document: using ElectoralRecordChunker (Household Co-Location Grouping)...")
        chunker = ElectoralRecordChunker(records_per_chunk=5)
    else:
        print(f"[*] Chunking {len(docs)} document units with StructureAwareParentChildChunker (preserving tables & parent context)...")
        chunker = StructureAwareParentChildChunker(rows_per_child=3, text_child_words=chunk_size)

    chunks: list[Chunk] = list(chunk_corpus(docs, chunker))
    if not chunks:
        # Fallback to fixed chunker if sentence chunker produced nothing
        chunks = list(chunk_corpus(docs, FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)))
    print(f"[+] Created {len(chunks)} chunks.")

    print("[*] Building SQLite FTS5 lexical index (slash-safe & scalable)...")
    fts5 = FTS5Index().build(chunks)

    print(f"[*] Connecting to Qdrant ({collection_name})...")
    embedder = Embedder("intfloat/multilingual-e5-small")
    # An explicit client lets callers use embedded/on-disk Qdrant when no
    # server is running — a desktop app should not require Docker.
    dense = DenseIndex(collection_name, embedder=embedder, client=qdrant_client)

    already_indexed = False
    if dense.client.collection_exists(collection_name):
        try:
            pt_count = dense.client.count(collection_name).count
            if pt_count == len(chunks) and not reindex:
                already_indexed = True
                print(f"[+] Reusing existing vector index ({pt_count} chunks). Use --reindex to force rebuild.")
        except Exception:
            already_indexed = False

    if not already_indexed:
        print(f"[*] Building vector index ({len(chunks)} chunks with INT8 Scalar Quantization)...")
        dense.recreate()
        dense.index(chunks, batch_size=32, show_progress=False)
        print(f"[+] Indexed {len(chunks)} chunks in Qdrant.")

    is_urdu = any("urdu" in (d.source or "").lower() or "bang-i-dara" in (d.source or "").lower() or "iqbal" in (d.source or "").lower() for d in docs)

    reranker = None
    if use_reranker:
        if reranker_model:
            chosen_reranker = reranker_model
        elif has_electoral:
            chosen_reranker = MULTILINGUAL_LIGHT
            print(f"[*] Electoral/Indic document detected: auto-selected multilingual reranker ({MULTILINGUAL_LIGHT})")
        elif is_urdu:
            # Urdu documents use dense multilingual-e5 + SQLite FTS5 for highest recall without English bias
            chosen_reranker = None
            print("[*] Urdu document detected: bypassing English cross-encoder in favor of multilingual dense + FTS5 retrieval.")
        else:
            chosen_reranker = DEFAULT_MODEL
        if chosen_reranker:
            print(f"[*] Initializing cross-encoder reranker ({chosen_reranker})...")
            reranker = CrossEncoderReranker(chosen_reranker)

    # Intent Router combining Page Lookup, Exact Entity match, Hybrid RRF, and neural reranker
    retriever = IntentRouter(
        lexical=fts5,
        dense=dense,
        reranker=reranker,
        candidate_limit=15,
    )

    # Threshold gate: abstain if top evidence is poor
    gate = ThresholdGate(threshold=threshold)

    # Resolve LLM backend
    chosen_backend = (backend or "auto").lower()
    if chosen_backend == "groq" or (chosen_backend == "auto" and os.environ.get("GROQ_API_KEY")):
        target_model = model or "llama-3.1-8b-instant"
        llm = GroqBackend(model=target_model)
        if not llm.available():
            print(f"[!] Groq backend selected but no valid GROQ_API_KEY found (starts with gsk_).")
            llm = get_llm(model=target_model)
    elif chosen_backend == "ollama" or chosen_backend == "auto":
        target_model = model
        ollama = OllamaBackend(model=target_model or "llama3.1:8b", timeout=600.0)
        if ollama.available() and target_model is None:
            # Auto-detect best/fastest model installed in Ollama
            try:
                installed = ollama.list_models()
                for fast_candidate in ("qwen2.5:1.5b", "llama3.2:1b", "llama3.2:3b", "qwen2.5:3b", "llama3.1:8b"):
                    matched = [m for m in installed if m.startswith(fast_candidate)]
                    if matched:
                        target_model = matched[0]
                        ollama = OllamaBackend(model=target_model, timeout=600.0)
                        break
            except Exception:
                pass
        llm = ollama if ollama.available() else get_llm(chosen_backend, model=target_model)
    else:
        llm = get_llm(chosen_backend, model=model)

    print(f"[+] Pipeline ready! Using LLM: {llm.name} ({llm.model})")
    pipeline = RAGPipeline(
        retriever=retriever,
        gate=gate,
        reranker=reranker,
        llm=llm,
        candidate_limit=15,
        evidence_limit=5,
        evidence_token_budget=1200,
        max_answer_tokens=250,
    )

    # Load the model now rather than inside the first question. A cold Ollama
    # pays a multi-GB weight load on its first request; leaving that inside the
    # request means the read timeout has to cover it, which is what produced
    # httpx.ReadTimeout on the first question.
    if hasattr(llm, "warmup"):
        print(f"[*] Warming up {llm.model} (loading weights)...")
        seconds = pipeline.warmup()
        print(f"[+] Model resident in {seconds:.1f}s.")

    return pipeline


def query_and_print(pipeline: RAGPipeline, question: str, stream: bool = True):
    """Execute query and print formatted answer, citations, and metrics."""
    print(f"\n=======================================================")
    print(f"Question: {question}")
    print(f"=======================================================")
    print("[*] Retrieving evidence and generating grounded answer...")

    start = time.perf_counter()
    tokens_streamed: list[str] = []

    def on_token(token: str):
        if not tokens_streamed:
            print("\n[Answer]:\n", end="", flush=True)
        tokens_streamed.append(token)
        sys.stdout.write(token)
        sys.stdout.flush()

    callback = on_token if stream else None
    result = pipeline.answer(question, stream_callback=callback)
    if tokens_streamed:
        print()
    total_time = (time.perf_counter() - start) * 1000

    print(f"\n[Decision]: {result.decision.value.upper()}")
    if result.abstention.score is not None:
        print(f"[Gate Score]: {result.abstention.score:.4f} (Threshold: {result.abstention.threshold:.4f})")

    if result.audit is not None:
        if not result.audit.is_clean:
            print(f"[!] Fact Audit Alert: Unverified figures in answer: {result.audit.unverified}")
        elif result.audit.verified:
            print(f"[+] Fact Audit: Grounded numbers verified: {result.audit.verified}")

    if result.abstained:
        print(f"\n[-] Pipeline ABSTAINED: {result.abstention.reason or 'Evidence below threshold or absent'}")
        if result.evidence:
            print(f"\n[Top Retrieved Chunk (Rejected)]:")
            top = result.evidence[0]
            print(f"  Score: {top.score:.4f} | Chunk ID: {top.chunk_id}")
            print(f"  Snippet: {top.text[:250]}...")
    elif not tokens_streamed:
        print(f"\n[Answer]:\n{result.answer}\n")

    if result.prompt_evidence:
        print("\n[Sources / Evidence Used]:")
        for idx, chunk in enumerate(result.prompt_evidence, 1):
            meta = chunk.metadata or {}
            page = getattr(chunk, "page", None) or meta.get("page") or (chunk.doc_id.split("#p")[-1].split("::")[0] if "#p" in chunk.doc_id else "N/A")
            print(f"  [{idx}] Doc: {chunk.doc_id} (Page {page})")
            print(f"      Snippet: {chunk.text[:180].replace(chr(10), ' ')}...\n")

    print("-------------------------------------------------------")
    latencies = [f"{k}: {v:.1f}ms" for k, v in result.latency_ms.items()]
    print(f"Timings: {' | '.join(latencies)} | Total: {total_time:.1f}ms")
    if result.llm:
        print(f"Tokens: in={result.llm.input_tokens}, out={result.llm.output_tokens}")
    print("=======================================================\n")


def main():
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Query any document using the RAG pipeline.")
    parser.add_argument("document", type=str, help="Path to document file (.pdf, .txt, .md)")
    parser.add_argument("-q", "--query", type=str, default=None, help="Single query to run")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "ollama", "groq", "openai", "anthropic"], help="LLM backend (auto, ollama, groq)")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. qwen2.5:1.5b, llama3.2:3b, llama-3.1-8b-instant)")
    parser.add_argument("--groq-key", type=str, default=None, help="Groq API Key (starts with gsk_)")
    parser.add_argument("--no-stream", action="store_true", help="Disable live token streaming")
    parser.add_argument("--threshold", type=float, default=-2.0, help="Abstention gate threshold")
    parser.add_argument("--ocr-lang", type=str, default="en", help="OCR language ('en', 'hi', etc.)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--reranker-model", type=str, default=None, help="Cross-encoder reranker model (default: ms-marco-MiniLM-L-6-v2, or multilingual e.g. cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)")
    parser.add_argument("--reindex", action="store_true", help="Force rebuilding vector index even if already exists")
    parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=40, help="Chunk overlap in words")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages to process from PDF (useful for quick testing)")
    parser.add_argument("--workers", type=int, default=1, help="Number of OCR worker processes (default: 1 sequential)")

    args = parser.parse_args()
    if args.groq_key:
        os.environ["GROQ_API_KEY"] = args.groq_key.strip()

    doc_path = Path(args.document)

    # Auto-detect Hindi language for electoral roll PDFs
    if args.ocr_lang == "en" and "HIN" in doc_path.name.upper():
        args.ocr_lang = "hi"
        print(f"[*] Auto-detected Hindi electoral document: set OCR language to 'hi'")

    # Auto-detect Urdu language
    if args.ocr_lang == "en" and any(k in str(doc_path).lower() for k in ["urdu", "bang-i-dara", "iqbal"]):
        args.ocr_lang = "urd"
        print(f"[*] Auto-detected Urdu document: set OCR language to 'urd'")

    docs = load_file(doc_path, ocr_lang=args.ocr_lang, max_pages=args.max_pages, workers=args.workers)
    collection = sanitize_collection_name(doc_path)

    pipeline = build_pipeline(
        docs=docs,
        collection_name=collection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        threshold=args.threshold,
        use_reranker=not args.no_rerank,
        reranker_model=args.reranker_model,
        backend=args.backend,
        model=args.model,
        reindex=args.reindex,
    )

    stream = not args.no_stream

    if args.query:
        query_and_print(pipeline, args.query, stream=stream)
    else:
        print("\n[+] Entering interactive mode. Type your questions below (or 'exit' / 'quit' to stop).\n")
        while True:
            try:
                q = input("Question > ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                query_and_print(pipeline, q, stream=stream)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
