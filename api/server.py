"""FastAPI backend for the RAG chat interface.

Serves the chat UI and exposes the pipeline over HTTP:

    GET  /                     the chat interface (the exported Next.js app
                               when web/out exists, else the legacy page)
    GET  /api/backends         which LLM backends are reachable, and their models
    GET  /api/browse           directory listing, for the folder picker
    GET  /api/health           liveness, plus model warm-up state
    POST /api/index            index a folder      (streams NDJSON progress)
    POST /api/chat             ask a question      (streams NDJSON tokens)

Both long operations stream NDJSON over a POST body rather than using
EventSource, which is GET-only. Indexing a scanned folder can take minutes of
OCR and generation on a local model can take minutes more, so neither can be a
plain request/response without the client appearing to hang. While the
pipeline is silent the stream carries a heartbeat every few seconds, and a
client that disconnects cancels its request.

Binds to 127.0.0.1 by default: `/api/browse` walks the local filesystem, so this
must not be exposed to a network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

ROOT = Path(__file__).resolve().parent.parent

# API keys (GROQ_API_KEY, ANTHROPIC_API_KEY) live in the project's .env. The
# eval scripts loaded it but the server never did, so hosted engines showed as
# unavailable in the UI however they were configured.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv is in requirements
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from console import use_utf8_console
from generation.llm import BACKENDS, OllamaBackend, available_backends, preferred_ollama_order

use_utf8_console()
logger = logging.getLogger("rag.api")

STATIC_DIR = Path(__file__).parent / "static"
# `pnpm build` in web/ exports the Next.js UI here.
WEB_OUT = ROOT / "web" / "out"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".csv", ".json", ".log"}

# Seconds between heartbeat events while the pipeline has nothing to say.
HEARTBEAT_SECONDS = 2.0
# OCR worker processes while indexing: all cores but one, as the CLI uses.
OCR_WORKERS = max(1, (os.cpu_count() or 2) - 1)
# Indexed folders kept in memory; each holds its lexical index and vectors.
MAX_SESSIONS = 3

# Directories that are never a user's data folder. Without this the picker
# offers __pycache__, venv and node_modules alongside real folders, and walking
# into venv/ lists thousands of package files as "indexable". The cache folders
# hold per-page OCR results as .json, which would otherwise be indexed as
# documents.
SKIP_DIRS = {
    "__pycache__", "venv", ".venv", "env", ".env", "node_modules", "site-packages",
    "dist", "build", ".git", ".cache", ".pytest_cache", ".mypy_cache", ".idea",
    ".vscode", "egg-info", ".ipynb_checkpoints", "indexes", ".ruff_cache",
    "cache", "ocr_cache",
}

# Project scaffolding that happens to carry a supported extension. These are
# configuration, not documents: counting requirements.txt as an indexable file
# makes an empty source tree look like it holds data.
SKIP_FILENAMES = {
    "requirements.txt", "requirements-dev.txt", "constraints.txt", "package.json",
    "package-lock.json", "tsconfig.json", "composer.json", "pyproject.toml",
    "setup.py", "setup.cfg", "cmakelists.txt", "license.txt", "license.md",
    "changelog.md", "contributing.md", "code_of_conduct.md", "makefile",
    ".gitignore", "manifest.json", "launch.json", "settings.json",
}
SKIP_SUFFIXES_EXACT = (".lock", ".egg-info")


def _is_data_file(path: Path) -> bool:
    """Is this a document a user would want indexed, rather than project config?"""
    if path.name.startswith("."):
        return False
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    if path.name.lower() in SKIP_FILENAMES:
        return False
    if path.name.lower().endswith(SKIP_SUFFIXES_EXACT):
        return False
    try:
        if path.stat().st_size == 0:
            return False
    except OSError:
        return False
    return True


def _is_browsable_dir(path: Path) -> bool:
    return not path.name.startswith(".") and path.name.lower() not in SKIP_DIRS


def _count_data_files(folder: Path, *, max_depth: int = 2, cap: int = 200) -> tuple[int, int]:
    """Return (direct, nested) indexable file counts for a folder.

    Nested counts matter because data usually sits a level or two down —
    browsing from a project root, `data/` would otherwise show nothing even
    though `data/fiqa/electoral/` holds every PDF. Depth and total are capped so
    listing a large tree stays responsive, and SKIP_DIRS keeps the walk out of
    virtualenvs and caches.
    """
    direct = nested = 0
    stack: list[tuple[Path, int]] = [(folder, 0)]
    while stack and (direct + nested) < cap:
        current, depth = stack.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if depth + 1 <= max_depth and _is_browsable_dir(entry):
                        stack.append((entry, depth + 1))
                elif _is_data_file(entry):
                    if depth == 0:
                        direct += 1
                    else:
                        nested += 1
            except (OSError, PermissionError):
                continue
    return direct, nested


def _data_files(folder: Path) -> list[Path]:
    """Indexable files under `folder`, recursively, skipping caches and envs.

    Recursive because data usually sits a level down; the picker already
    counts nested files, so indexing only the top level made those vanish.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if _is_browsable_dir(Path(d)))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if _is_data_file(path):
                found.append(path)
    return found


def _friendly(exc: Exception) -> str:
    """Turn the failures users actually hit into something actionable."""
    text = str(exc)
    lowered = text.lower()
    if "11434" in text or "ollama serve" in lowered:
        return "Ollama is not running. Start it with `ollama serve`, or choose another engine."
    if "401" in text and "groq" in lowered:
        return "Groq rejected the API key. Check GROQ_API_KEY in .env, then restart the server."
    return text


# --------------------------------------------------------------------------
# Pipeline thread, warm-up and sessions
# --------------------------------------------------------------------------


# All pipeline work runs on ONE dedicated thread. Some stores (SQLite-backed
# ones, including the FTS5 index) cannot cross threads, and on a laptop
# concurrent OCR and embedding thrash rather than parallelise. The cost is
# that one request waits for another, so the stream says so, and a
# disconnected client cancels its work instead of holding the thread.
_PIPELINE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-pipeline")
_pending = 0
_pending_lock = threading.Lock()

_warm: dict[str, Any] = {"state": "cold", "seconds": None, "detail": ""}


def _warm_up() -> None:
    """Pay the heavy imports and model loads at startup, not in the first request.

    Importing sentence-transformers alone can take a minute on a cold disk;
    inside the first "Index folder" that looked like a hung UI.
    """
    started = time.perf_counter()
    _warm["state"] = "warming"
    try:
        import ask  # noqa: F401 - torch, sentence-transformers, the pipeline
        from rerank.cross_encoder import DEFAULT_MODEL, MULTILINGUAL_LIGHT, get_reranker
        from retrieval.embedder import MULTILINGUAL_MODEL, get_embedder

        get_embedder(MULTILINGUAL_MODEL).model
        for name in (MULTILINGUAL_LIGHT, DEFAULT_MODEL):
            get_reranker(name).model
        _warm.update(state="ready", seconds=round(time.perf_counter() - started, 1))
    except Exception as exc:
        logger.exception("Warm-up failed")
        _warm.update(state="error", detail=str(exc))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if os.environ.get("RAG_SKIP_WARMUP") != "1":
        _PIPELINE.submit(_warm_up)
    yield


app = FastAPI(title="RAG Chat", lifespan=_lifespan)

# One pipeline per (folder, backend, model), most recently used last. Rebuilding
# costs chunking at least; the OCR cache and stored vectors make it cheap.
_sessions: "OrderedDict[str, Any]" = OrderedDict()
_sessions_lock = threading.Lock()


def _remember(key: str, pipeline: Any) -> None:
    with _sessions_lock:
        _sessions[key] = pipeline
        _sessions.move_to_end(key)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class IndexRequest(BaseModel):
    folder: str
    backend: str = "auto"
    model: Optional[str] = None
    ocr_lang: str = "auto"
    max_files: int = 25
    max_pages: Optional[int] = None


class ChatRequest(BaseModel):
    session_id: str
    question: str


# --------------------------------------------------------------------------
# Backend / model discovery
# --------------------------------------------------------------------------


def _ollama_models() -> list[dict]:
    try:
        details = OllamaBackend().list_model_details()
    except Exception:
        return []
    by_name = {m["name"]: m for m in details}
    return [by_name[name] for name in preferred_ollama_order(list(by_name))]


@app.get("/api/backends")
def list_backends() -> dict:
    """Report every backend and the models it can actually serve.

    Availability is probed live so the picker never offers a backend whose
    server is down or whose API key is missing. One backend is marked
    `recommended`: the fastest that can answer here.
    """
    reachable = available_backends()
    ollama = _ollama_models() if reachable.get("ollama") else []

    catalogue = [
        {
            "id": "groq",
            "label": "Groq API (fastest)",
            "available": reachable.get("groq", False),
            "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
            "hint": "set GROQ_API_KEY in .env",
        },
        {
            "id": "ollama",
            "label": "Ollama (local)",
            "available": reachable.get("ollama", False),
            "models": [m["name"] for m in ollama],
            "sizes": {m["name"]: m["size"] for m in ollama},
            "hint": "ollama serve" if not reachable.get("ollama") else "",
        },
        {
            "id": "openai",
            "label": "Local OpenAI-compatible server",
            "available": reachable.get("openai", False),
            "models": [],
            "hint": "llama.cpp / LM Studio / vLLM on :8080",
        },
        {
            "id": "anthropic",
            "label": "Anthropic API",
            "available": reachable.get("anthropic", False),
            "models": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
            "hint": "set ANTHROPIC_API_KEY in .env",
        },
        {
            "id": "stub",
            "label": "Stub (no model — pipeline test)",
            "available": True,
            "models": ["stub"],
            "hint": "quotes the top evidence; for checking retrieval only",
        },
    ]
    usable = [b for b in catalogue if b["available"] and b["id"] != "stub"
              and (b["models"] or b["id"] == "openai")]
    recommended = usable[0]["id"] if usable else None
    for backend in catalogue:
        backend["recommended"] = backend["id"] == recommended
    return {"backends": [b for b in catalogue if b["id"] in BACKENDS]}


# --------------------------------------------------------------------------
# Folder picker
# --------------------------------------------------------------------------


@app.get("/api/browse")
def browse(path: Optional[str] = None) -> dict:
    """List sub-directories and indexable files under `path`."""
    if path:
        target = Path(path).expanduser()
    else:
        target = Path.cwd()
    try:
        target = target.resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail=f"No such folder: {path}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a folder: {target}")

    dirs, files = [], []
    try:
        entries = list(target.iterdir())
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")

    for entry in sorted(entries, key=lambda p: p.name.lower()):
        try:
            if entry.is_dir():
                if not _is_browsable_dir(entry):
                    continue
                # Surfacing the count is what lets someone find the data folder
                # without opening every directory in turn.
                direct, nested = _count_data_files(entry)
                dirs.append({"name": entry.name, "path": str(entry),
                             "data_files": direct, "nested_files": nested})
            elif _is_data_file(entry):
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                files.append({"name": entry.name, "size": size,
                              "suffix": entry.suffix.lower()})
        except (PermissionError, OSError):
            continue

    return {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "dirs": dirs,
        "files": files,
        "indexable": len(files) + sum(d["data_files"] + d["nested_files"] for d in dirs),
    }


# --------------------------------------------------------------------------
# NDJSON streaming
# --------------------------------------------------------------------------


# Every hop between here and the browser must pass lines through as they are
# written: no caching, no transformation (compression buffers), no buffering
# (nginx and similar honour X-Accel-Buffering).
_NDJSON_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


class ClientGone(Exception):
    """The browser disconnected; its request is abandoned."""


def _ndjson(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def _stream(request: Request, work) -> StreamingResponse:
    return StreamingResponse(_stream_worker(request, work),
                             media_type="application/x-ndjson", headers=_NDJSON_HEADERS)


async def _stream_worker(request: Request, work) -> AsyncIterator[str]:
    """Run `work(emit)` on the pipeline thread, streaming what it emits.

    - While the work is silent (OCR, a model load, a CPU model's prompt
      evaluation) a heartbeat goes out every HEARTBEAT_SECONDS, so neither the
      UI nor a proxy mistakes a slow step for a dead connection.
    - When the client disconnects, the next `emit` raises ClientGone inside the
      worker: generation stops at the next token and the single pipeline thread
      is free again. An abandoned answer used to keep generating, and every
      later question queued silently behind it.
    """
    global _pending
    events: queue.Queue = queue.Queue()
    finished = object()
    cancelled = threading.Event()
    stage = {"name": "queued"}

    def emit(event: dict) -> None:
        if cancelled.is_set():
            raise ClientGone()
        if event.get("stage"):
            stage["name"] = event["stage"]
        events.put(event)

    def run() -> None:
        global _pending
        try:
            if not cancelled.is_set():
                stage["name"] = "starting"
                work(emit)
        except ClientGone:
            logger.info("Client disconnected; its request was abandoned.")
        except Exception as exc:
            logger.exception("Request failed")
            events.put({"type": "error", "message": _friendly(exc),
                        "detail": traceback.format_exc(limit=3)})
        finally:
            with _pending_lock:
                _pending -= 1
            events.put(finished)

    with _pending_lock:
        ahead = _pending
        _pending += 1
    _PIPELINE.submit(run)

    if _warm["state"] == "warming":
        yield _ndjson({"type": "status", "stage": "warming",
                       "message": "Loading the embedding and ranking models (first start only)..."})
    if ahead:
        yield _ndjson({"type": "status", "stage": "queued",
                       "message": f"Waiting for {ahead} earlier request(s) to finish..."})

    started = time.perf_counter()
    try:
        while True:
            try:
                event = await asyncio.to_thread(events.get, True, HEARTBEAT_SECONDS)
            except queue.Empty:
                if await request.is_disconnected():
                    return
                yield _ndjson({"type": "heartbeat", "stage": stage["name"],
                               "elapsed": round(time.perf_counter() - started, 1)})
                continue
            if event is finished:
                return
            yield _ndjson(event)
    finally:
        # Runs on normal completion and when the client goes away (the
        # response task is cancelled at the await above).
        cancelled.set()


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def _session_key(request: IndexRequest) -> str:
    return f"{request.folder}|{request.backend}|{request.model or 'default'}"


def _collection_name(folder: Path) -> str:
    """Stable vector-collection name for a folder.

    Derived with sha256 rather than `hash()`: string hashing is salted per
    process, so the old name changed on every server restart — each restart
    re-embedded the whole folder and orphaned the previous collection.
    """
    slug = "".join(ch if ch.isalnum() else "_" for ch in folder.name)[:24]
    digest = hashlib.sha256(str(folder).encode("utf-8")).hexdigest()[:8]
    return f"folder_{slug}_{digest}"


def _build(request: IndexRequest, emit) -> None:
    # Imported lazily: these pull in torch and sentence-transformers, and the
    # server should start instantly even when no model is installed.
    import ask

    folder = Path(request.folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a folder: {folder}")

    candidates = _data_files(folder)[: request.max_files]
    if not candidates:
        raise ValueError(
            f"No indexable files in {folder}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    total = len(candidates)

    def progress(message: str, stage: str, current: Optional[int] = None) -> None:
        emit({"type": "progress", "stage": stage, "message": message,
              "current": current, "total": total})

    ocr_lang = ask.detect_ocr_lang(folder, request.ocr_lang)
    progress(f"Found {total} file(s) in {folder.name} · OCR language {ocr_lang}", "scan", 0)

    # One extractor for the whole folder: it owns the OCR engine, and building
    # one per file reloaded the model for every PDF.
    extractor = ask.make_extractor(ocr_lang=ocr_lang, workers=OCR_WORKERS)

    documents = []
    for position, path in enumerate(candidates, 1):
        progress(f"Extracting {path.name}", "extract", position - 1)
        started = time.perf_counter()
        try:
            docs = ask.load_file(path, ocr_lang=ocr_lang, max_pages=request.max_pages,
                                 workers=OCR_WORKERS, extractor=extractor)
        except Exception as exc:
            # One unreadable file must not abandon an expensive multi-file run.
            progress(f"Skipped {path.name}: {exc}", "extract", position)
            continue
        documents.extend(docs)
        progress(f"{path.name}: {len(docs)} page(s) in {time.perf_counter() - started:.1f}s",
                 "extract", position)

    if not documents:
        raise ValueError("No text could be extracted from any file.")

    pipeline = ask.build_pipeline(
        documents,
        _collection_name(folder),
        backend=request.backend,
        model=request.model,
        progress=lambda message: progress(message.lstrip("[*+!] "), "index", total),
    )

    key = _session_key(request)
    _remember(key, pipeline)

    emit({
        "type": "done",
        "session_id": key,
        "documents": len(documents),
        "files": total,
        "folder": str(folder),
        "backend": getattr(pipeline.llm, "name", request.backend),
        "model": getattr(pipeline.llm, "model", request.model or ""),
    })


@app.post("/api/index")
def index_folder(body: IndexRequest, request: Request) -> StreamingResponse:
    return _stream(request, lambda emit: _build(body, emit))


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


def _answer(request: ChatRequest, emit) -> None:
    with _sessions_lock:
        pipeline = _sessions.get(request.session_id)
        if pipeline is not None:
            _sessions.move_to_end(request.session_id)
    if pipeline is None:
        raise ValueError("This folder is no longer indexed. Index it again.")

    question = request.question.strip()
    if not question:
        raise ValueError("Empty question.")

    result = pipeline.answer(
        question,
        stream_callback=lambda token: emit({"type": "token", "text": token}),
        on_stage=lambda name, message: emit({"type": "status", "stage": name, "message": message}),
    )

    citations = []
    for index, scored in enumerate(result.prompt_evidence, 1):
        citations.append({
            "n": index,
            "citation": scored.citation(),
            "page": scored.page,
            "source": Path(scored.source).name if scored.source else "",
            "preview": " ".join(scored.text.split())[:240],
        })

    emit({
        "type": "done",
        "answer": result.answer,
        "abstained": result.abstained,
        "decision": result.decision.value,
        "reason": result.abstention.reason,
        "grounded": result.grounded,
        "citations": citations,
        "metrics": result.to_dict(),
    })


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    return _stream(request, lambda emit: _answer(body, emit))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "sessions": len(_sessions), "warm": dict(_warm)}


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------


if (WEB_OUT / "index.html").exists():
    # The exported Next.js UI, served by this process: one port, no proxy.
    # Mounted last, so every /api route above still wins.
    app.mount("/", StaticFiles(directory=str(WEB_OUT), html=True), name="web")
else:
    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run(host: str = "127.0.0.1", port: Optional[int] = None) -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the RAG chat server.")
    parser.add_argument("--host", type=str, default=host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=port or int(os.environ.get("PORT", "8000")), help="Port to bind to")
    args, _ = parser.parse_known_args()

    print(f"[*] Starting RAG Chat server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
