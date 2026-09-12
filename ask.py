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
import platform
import re
import sys
import time
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from generation.abstention import ThresholdGate
from console import use_utf8_console
from generation.llm import (
    FallbackBackend,
    GroqBackend,
    OllamaBackend,
    get_llm,
    preferred_ollama_order,
)
from generation.pipeline import RAGPipeline
from ingestion.chunking import FixedSizeChunker, StructureAwareParentChildChunker
from ingestion.documents import Chunk, Document
from ingestion.pdf_extractor import PDFExtractor
from ingestion.pipeline import chunk_corpus
from rerank.cross_encoder import DEFAULT_MODEL, MULTILINGUAL_LIGHT, get_reranker
from retrieval.dense import make_dense_index
from retrieval.embedder import MULTILINGUAL_MODEL, get_embedder
from retrieval.fts5_index import FTS5Index
from retrieval.router import IntentRouter

# Path fragments of Urdu (InPage / Nastaliq) sources.
_URDU_MARKERS = ("urdu", "bang-i-dara", "iqbal")
# "-HIN-" in the Bihar electoral roll file names. A bare substring test also
# matched ordinary words such as "WITHIN".
_HINDI_FILE = re.compile(r"(?:^|[^A-Z])HIN(?:[^A-Z]|$)")


def detect_ocr_lang(doc_path: Path, requested: str = "auto") -> str:
    """OCR language for a file or folder; shared by the CLI and the API.

    "auto" (and the old default "en") read the file names. Hindi electoral
    rolls always need "hin+eng": the names are Devanagari, the EPIC IDs Latin.
    The API used to OCR every folder as English unless told otherwise, which
    turned the Hindi rolls into noise.
    """
    requested = (requested or "auto").lower()
    names = [p.name for p in doc_path.rglob("*.pdf")] if doc_path.is_dir() else [doc_path.name]
    if requested in ("auto", "en", "hi") and any(_HINDI_FILE.search(n.upper()) for n in names):
        return "hin+eng"
    if requested == "hi":
        return "hin+eng"
    if requested in ("auto", "en") and any(k in str(doc_path).lower() for k in _URDU_MARKERS):
        return "urd"
    return "en" if requested == "auto" else requested


def make_extractor(
    *,
    ocr_lang: str = "en",
    ocr_engine: str = "auto",
    workers: int = 1,
    use_ocr_cache: bool = True,
    clear_cache: bool = False,
) -> PDFExtractor:
    """Build the PDF extractor for one ingestion run.

    Build it once and pass it to every `load_file` call: it owns the OCR
    engine, and a fresh extractor per file reloaded the model (~5 s) for every
    PDF in a folder.
    """
    provider = None
    if ocr_engine.lower() == "tesseract":
        from ingestion.ocr import TesseractOCRProvider
        provider = TesseractOCRProvider(lang=ocr_lang)

    extractor = PDFExtractor(
        ocr_provider=provider,
        ocr_lang=ocr_lang,
        use_cache=use_ocr_cache,
        workers=workers,
    )
    if clear_cache and hasattr(extractor.cache, "clear"):
        extractor.cache.clear()
    return extractor


def load_file(
    file_path: Path,
    *,
    ocr_lang: str = "en",
    ocr_engine: str = "auto",
    max_pages: Optional[int] = None,
    workers: int = 1,
    use_ocr_cache: bool = True,
    clear_cache: bool = False,
    extractor: Optional[PDFExtractor] = None,
) -> list[Document]:
    """Load and extract text from a file or directory (.pdf, .txt, .md, etc.)."""
    if not file_path.exists():
        raise FileNotFoundError(f"File or directory not found: {file_path}")

    if file_path.is_dir():
        print(f"[*] Discovering documents in directory: {file_path}...")
        supported_exts = {".pdf", ".txt", ".md", ".csv", ".json", ".log"}
        all_files = sorted([p for p in file_path.rglob("*") if p.is_file() and p.suffix.lower() in supported_exts])
        if not all_files:
            raise ValueError(f"No supported document files ({', '.join(supported_exts)}) found in directory: {file_path}")
        print(f"[+] Found {len(all_files)} documents to ingest in {file_path.name}/.")
        extractor = extractor or make_extractor(
            ocr_lang=ocr_lang,
            ocr_engine=ocr_engine,
            workers=workers,
            use_ocr_cache=use_ocr_cache,
            clear_cache=clear_cache,
        )
        all_docs = []
        for idx, sub_path in enumerate(all_files, 1):
            print(f"[{idx}/{len(all_files)}] Ingesting file: {sub_path.name}")
            sub_docs = load_file(
                sub_path,
                ocr_lang=ocr_lang,
                ocr_engine=ocr_engine,
                max_pages=max_pages,
                workers=workers,
                use_ocr_cache=use_ocr_cache,
                extractor=extractor,
            )
            # Ensure doc_ids are distinct across different files in directory
            for d in sub_docs:
                if "#p" not in d.doc_id and not d.doc_id.startswith(sub_path.stem):
                    d.doc_id = f"{sub_path.stem}_{d.doc_id}"
            all_docs.extend(sub_docs)
        print(f"[+] Total documents/pages ingested from directory: {len(all_docs)}")
        return all_docs

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        page_info = f" (max_pages={max_pages})" if max_pages else ""
        print(f"[*] Extracting text from PDF: {file_path.name}{page_info} (OCR engine: {ocr_engine}, lang: {ocr_lang}, workers: {workers}, cache: {use_ocr_cache})...")
        start = time.perf_counter()
        
        extractor = extractor or make_extractor(
            ocr_lang=ocr_lang,
            ocr_engine=ocr_engine,
            workers=workers,
            use_ocr_cache=use_ocr_cache,
            clear_cache=clear_cache,
        )

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


# Part of every collection name. Bump it whenever chunk payloads change shape:
# index reuse is decided by chunk count alone, so an old collection would
# otherwise be served as-is.
INDEX_VERSION = "v5"


def sanitize_collection_name(file_path: Path) -> str:
    """Ensure collection name meets Qdrant conventions and is content-addressed."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", file_path.stem)
    try:
        if file_path.is_dir():
            files = sorted([str(p.relative_to(file_path)) for p in file_path.rglob("*") if p.is_file()])
            h = hashlib.sha256(("::".join(files)).encode("utf-8")).hexdigest()[:8]
            return f"dir_{INDEX_VERSION}_{clean[:15]}_{h}"
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:8]
    except Exception:
        digest = "default"
    return f"doc_{INDEX_VERSION}_{clean[:17]}_{digest}"


_MANIFEST_DIR = Path(__file__).resolve().parent / ".cache" / "index_manifests"
_DEVANAGARI = re.compile("[ऀ-ॿ]")
_ARABIC_SCRIPT = re.compile("[؀-ۿ]")
_LATIN = re.compile("[A-Za-z]")
# Share of Devanagari/Arabic letters above which the multilingual reranker is
# worth its cost. Hindi rolls are mostly Devanagari; the Master Plan is <1%.
_NON_LATIN_SHARE = 0.15


def _corpus_fingerprint(chunks: list[Chunk], embedder_name: str) -> str:
    """Hash of exactly what gets embedded. Equal hashes mean stored vectors are still valid."""
    digest = hashlib.sha256(f"{INDEX_VERSION}|{embedder_name}".encode("utf-8"))
    for c in chunks:
        digest.update(f"{c.chunk_id}\x00{c.text}\x01".encode("utf-8"))
    return digest.hexdigest()


def _stored_fingerprint(collection: str) -> Optional[str]:
    try:
        return (_MANIFEST_DIR / f"{collection}.sha256").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _store_fingerprint(collection: str, fingerprint: str) -> None:
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (_MANIFEST_DIR / f"{collection}.sha256").write_text(fingerprint, encoding="utf-8")


def _choose_reranker(docs: list[Document]) -> Optional[str]:
    """Cross-encoder for this corpus, judged from its text rather than file names.

    Devanagari (with or without Latin) needs the multilingual model, which was
    trained on mMARCO's Hindi. No available cross-encoder covers Urdu, so an
    Urdu-only corpus ranks with multilingual dense + FTS5 alone.
    """
    # Judged by share of letters, not presence: the Master Plan quotes a few
    # Hindi words, and "any Devanagari" put it on the 12-layer multilingual
    # model, 2-4x slower per question than the English one.
    indic = sum(len(_DEVANAGARI.findall(d.text)) for d in docs)
    arabic = sum(len(_ARABIC_SCRIPT.findall(d.text)) for d in docs)
    latin = sum(len(_LATIN.findall(d.text)) for d in docs)
    share = (indic + arabic) / max(indic + arabic + latin, 1)
    if share < _NON_LATIN_SHARE:
        return DEFAULT_MODEL
    if arabic > indic:
        return None
    return MULTILINGUAL_LIGHT


def _pick_ollama(model: Optional[str]) -> Optional[OllamaBackend]:
    """Ollama serving `model`, or its preferred installed model; None if Ollama is down."""
    probe = OllamaBackend(model=model or "llama3.1:8b", timeout=600.0)
    if not probe.available():
        return None
    if model:
        return probe
    try:
        installed = preferred_ollama_order(probe.list_models())
    except Exception:
        return None
    return OllamaBackend(model=installed[0], timeout=600.0) if installed else None


def resolve_llm(backend: Optional[str], model: Optional[str], *, progress=print):
    """The answer engine for `backend`.

    Groq, when chosen or when "auto" finds a key, gets the local Ollama model
    as a fallback: a rate limit or a dropped connection then costs speed rather
    than the answer.
    """
    chosen = (backend or "auto").lower()
    if chosen == "groq" or (chosen == "auto" and os.environ.get("GROQ_API_KEY")):
        groq = GroqBackend(model=model or "llama-3.1-8b-instant")
        if groq.available():
            fallback = _pick_ollama(None)
            return FallbackBackend(groq, fallback) if fallback else groq
        progress("[!] Groq selected but GROQ_API_KEY is not set to a valid gsk_... key; trying a local engine.")
        chosen = "auto"
        model = None
    if chosen in ("ollama", "auto"):
        ollama = _pick_ollama(model)
        if ollama is not None:
            return ollama
    return get_llm(chosen, model=model)


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
    dense_index=None,
    progress=print,
) -> RAGPipeline:
    """Build chunks, indexes, retriever, and RAGPipeline.

    `progress` receives one line per step: the CLI prints them, the API streams
    them to the browser.
    """
    # One chunker for every page: it hands each electoral page to the voter-card
    # chunker itself. Picking a single chunker for the whole corpus sent a mixed
    # folder's Master Plan through the voter-card chunker and lost its tables.
    progress(f"[*] Chunking {len(docs)} document units (tables, sections and voter records kept whole)...")
    chunker = StructureAwareParentChildChunker(rows_per_child=3, text_child_words=chunk_size)
    chunks: list[Chunk] = list(chunk_corpus(docs, chunker))
    if not chunks:
        # Fallback to fixed chunker if the structure-aware chunker produced nothing
        chunks = list(chunk_corpus(docs, FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)))
    progress(f"[+] Created {len(chunks)} chunks.")

    progress("[*] Building SQLite FTS5 lexical index...")
    fts5 = FTS5Index().build(chunks)

    embedder = get_embedder(MULTILINGUAL_MODEL)
    dense = dense_index or make_dense_index(collection_name, embedder)
    # Reuse is decided by a hash of the chunk texts, not a chunk count: a
    # changed file with the same number of chunks must be re-embedded, and an
    # unchanged folder re-opened after a restart must not be.
    fingerprint = _corpus_fingerprint(chunks, embedder.model_name)
    if (not reindex and dense.exists() and dense.count() == len(chunks)
            and _stored_fingerprint(collection_name) == fingerprint):
        progress(f"[+] Reusing the stored vector index ({len(chunks)} chunks). Use --reindex to rebuild.")
    else:
        progress(f"[*] Embedding {len(chunks)} chunks into {type(dense).__name__}...")
        started = time.perf_counter()
        dense.recreate()
        dense.index(chunks, show_progress=False)
        _store_fingerprint(collection_name, fingerprint)
        progress(f"[+] Embedded {len(chunks)} chunks in {time.perf_counter() - started:.1f}s.")

    reranker = None
    if use_reranker:
        chosen_reranker = reranker_model or _choose_reranker(docs)
        if chosen_reranker:
            progress(f"[*] Loading reranker {chosen_reranker}...")
            reranker = get_reranker(chosen_reranker)
        else:
            progress("[*] Urdu corpus: no cross-encoder covers Urdu; ranking with multilingual dense + FTS5.")

    # Intent Router combining Page Lookup, Exact Entity match, Hybrid RRF, and neural reranker
    retriever = IntentRouter(
        lexical=fts5,
        dense=dense,
        reranker=reranker,
        candidate_limit=15,
    )

    # Threshold gate: abstain if top evidence is poor
    gate = ThresholdGate(threshold=threshold)

    llm = resolve_llm(backend, model, progress=progress)
    progress(f"[+] Pipeline ready. Answer engine: {llm.name} ({llm.model})")

    # Prompt size is the main lever on answer latency. A CPU model evaluates the
    # prompt at tens of tokens per second, so it gets the small local preset; a
    # hosted engine reads a few thousand tokens in well under a second.
    if getattr(llm, "name", "") in ("ollama", "openai"):
        pipeline = RAGPipeline.for_local_model(
            retriever, gate, reranker=reranker, llm=llm, candidate_limit=15,
            roster_token_budget=2000,
        )
    else:
        pipeline = RAGPipeline(
            retriever, gate, reranker=reranker, llm=llm, candidate_limit=15,
            evidence_limit=6, evidence_token_budget=2500, max_answer_tokens=400,
            roster_token_budget=4000,
        )

    # Load the model now rather than inside the first question. A cold Ollama
    # pays a multi-GB weight load on its first request; leaving that inside the
    # request means the read timeout has to cover it, which is what produced
    # httpx.ReadTimeout on the first question.
    if hasattr(llm, "warmup"):
        progress(f"[*] Warming up {llm.model} (loading weights)...")
        seconds = pipeline.warmup()
        progress(f"[+] Model resident in {seconds:.1f}s.")

    return pipeline


def query_and_print(
    pipeline: RAGPipeline,
    question: str,
    stream: bool = True,
    target_lang: Optional[str] = None,
):
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
    result = pipeline.answer(question, stream_callback=callback, target_lang=target_lang)
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
    default_workers = max(1, (os.cpu_count() or 4) - 1)
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Number of OCR worker processes (default: {default_workers}, reserving 1 core for SSH/system)")
    parser.add_argument("--no-ocr-cache", "--reextract", action="store_true", help="Bypass OCR disk cache and force fresh extraction")
    parser.add_argument("--clear-cache", action="store_true", help="Purge disk extraction cache before extracting")
    parser.add_argument("--answer-lang", choices=["auto", "en", "hi", "ur"], default="auto", help="Response language override (default: auto)")
    parser.add_argument("--ocr-engine", choices=["auto", "paddle", "tesseract"], default="auto", help="OCR engine to use (default: auto)")

    args = parser.parse_args()
    if args.ocr_engine == "paddle":
        if sys.platform.startswith("linux") and platform.machine().lower() in ("aarch64", "arm64"):
            print("[!] WARNING: PaddlePaddle has a known C++ ABI crash (SIGSEGV) on Linux ARM64. Falling back to native Tesseract.")
            args.ocr_engine = "tesseract"
            os.environ["ENABLE_PADDLEOCR"] = "0"
        else:
            os.environ["ENABLE_PADDLEOCR"] = "1"
    elif args.ocr_engine == "tesseract":
        os.environ["ENABLE_PADDLEOCR"] = "0"

    if args.groq_key:
        os.environ["GROQ_API_KEY"] = args.groq_key.strip()

    doc_path = Path(args.document)

    if args.clear_cache:
        import shutil
        cache_dir = Path(__file__).resolve().parent / ".cache" / "extraction"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("[+] Purged all extraction disk caches (.cache/extraction).")
        args.reindex = True

    detected = detect_ocr_lang(doc_path, args.ocr_lang)
    if detected != args.ocr_lang:
        print(f"[*] OCR language: '{detected}' (auto-detected from the file names)")
    args.ocr_lang = detected

    docs = load_file(
        doc_path,
        ocr_lang=args.ocr_lang,
        ocr_engine=args.ocr_engine,
        max_pages=args.max_pages,
        workers=args.workers,
        use_ocr_cache=not args.no_ocr_cache,
        clear_cache=args.clear_cache,
    )
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
        query_and_print(pipeline, args.query, stream=stream, target_lang=args.answer_lang)
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
                query_and_print(pipeline, q, stream=stream, target_lang=args.answer_lang)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
