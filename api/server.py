"""FastAPI backend for the RAG chat interface.

Serves a single-page chat UI and exposes the pipeline over HTTP:

    GET  /                     the chat interface
    GET  /api/backends         which LLM backends are reachable, and their models
    GET  /api/browse           directory listing, for the folder picker
    POST /api/index            index a folder      (streams NDJSON progress)
    POST /api/chat             ask a question      (streams NDJSON tokens)

Both long operations stream NDJSON over a POST body rather than using
EventSource, which is GET-only. Indexing a scanned folder can take minutes of
OCR and generation on a local model can take minutes more, so neither can be a
plain request/response without the client appearing to hang.

Binds to 127.0.0.1 by default: `/api/browse` walks the local filesystem, so this
must not be exposed to a network.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from console import use_utf8_console
from generation.llm import BACKENDS, OllamaBackend, available_backends

try:
    from ingestion.ocr import NoOCREngineAvailable as NoOCREngineAvailableType
except Exception:  # pragma: no cover - older ocr module
    NoOCREngineAvailableType = None

use_utf8_console()

STATIC_DIR = Path(__file__).parent / "static"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".csv", ".json", ".log"}

# Directories that are never a user's data folder. Without this the picker
# offers __pycache__, venv and node_modules alongside real folders, and walking
# into venv/ lists thousands of package files as "indexable".
SKIP_DIRS = {
    "__pycache__", "venv", ".venv", "env", ".env", "node_modules", "site-packages",
    "dist", "build", ".git", ".cache", ".pytest_cache", ".mypy_cache", ".idea",
    ".vscode", "egg-info", ".ipynb_checkpoints", "indexes", ".ruff_cache",
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
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# On-disk embedded Qdrant, used when no server answers. Keeps the app usable
# without Docker while still persisting vectors between runs.
EMBEDDED_QDRANT_PATH = Path(__file__).resolve().parent.parent / ".cache" / "qdrant"


def _qdrant_client():
    """Return (client, description). Prefers a running server, else embedded."""
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2.0)
        client.get_collections()
        return client, f"Qdrant server at {QDRANT_URL}"
    except Exception:
        EMBEDDED_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        return (QdrantClient(path=str(EMBEDDED_QDRANT_PATH)),
                f"embedded Qdrant at {EMBEDDED_QDRANT_PATH.name}/ (no server running)")


def _friendly(exc: Exception) -> str:
    """Turn the failures users actually hit into something actionable."""
    text = str(exc)
    lowered = text.lower()
    if "connection refused" in lowered and "6333" in text:
        return ("Qdrant is not reachable and embedded mode could not start. "
                "Run `docker compose up -d`, or free the .cache/qdrant folder.")
    if "11434" in text or "ollama serve" in lowered:
        return "Ollama is not running. Start it with `ollama serve`, then re-index."
    if isinstance(exc, NoOCREngineAvailableType) if NoOCREngineAvailableType else False:
        return text
    return text

app = FastAPI(title="RAG Chat")

# One pipeline per (folder, backend, model). Rebuilding is expensive — the OCR
# cache makes re-extraction cheap, but chunking and embedding are not free.
_sessions: dict[str, Any] = {}
_sessions_lock = threading.Lock()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class IndexRequest(BaseModel):
    folder: str
    backend: str = "auto"
    model: Optional[str] = None
    ocr_lang: str = "en"
    max_files: int = 25
    max_pages: Optional[int] = None


class ChatRequest(BaseModel):
    session_id: str
    question: str


# --------------------------------------------------------------------------
# Backend / model discovery
# --------------------------------------------------------------------------


def _ollama_models() -> list[str]:
    try:
        return OllamaBackend().list_models()
    except Exception:
        return []


@app.get("/api/backends")
def list_backends() -> dict:
    """Report every backend and the models it can actually serve.

    Availability is probed live so the picker never offers a backend whose
    server is down or whose API key is missing.
    """
    reachable = available_backends()
    ollama_models = _ollama_models() if reachable.get("ollama") else []

    catalogue = [
        {
            "id": "ollama",
            "label": "Ollama (local)",
            "available": reachable.get("ollama", False),
            "models": ollama_models,
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
            "id": "groq",
            "label": "Groq API",
            "available": reachable.get("groq", False),
            "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile",
                       "gemma2-9b-it"],
            "hint": "set GROQ_API_KEY",
        },
        {
            "id": "anthropic",
            "label": "Anthropic API",
            "available": reachable.get("anthropic", False),
            "models": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
            "hint": "set ANTHROPIC_API_KEY",
        },
        {
            "id": "stub",
            "label": "Stub (no model — pipeline test)",
            "available": True,
            "models": ["stub"],
            "hint": "quotes the top evidence; for checking retrieval only",
        },
    ]
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
        "indexable": len(files),
    }


# --------------------------------------------------------------------------
# NDJSON streaming helper
# --------------------------------------------------------------------------


def _ndjson(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def _stream_worker(work) -> Iterator[str]:
    """Run `work(emit)` on a thread, yielding whatever it emits.

    The pipeline is synchronous and CPU-bound; running it inline would block the
    event loop and stall every other request, including the browser's own
    progress rendering.
    """
    events: queue.Queue = queue.Queue()
    sentinel = object()

    def emit(event: dict) -> None:
        events.put(event)

    def run() -> None:
        try:
            work(emit)
        except Exception as exc:
            events.put({"type": "error", "message": _friendly(exc),
                        "detail": traceback.format_exc(limit=3)})
        finally:
            events.put(sentinel)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    while True:
        event = events.get()
        if event is sentinel:
            return
        yield _ndjson(event)


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def _session_key(request: IndexRequest) -> str:
    return f"{request.folder}|{request.backend}|{request.model or 'default'}"


def _build(request: IndexRequest, emit) -> None:
    # Imported lazily: these pull in torch and sentence-transformers, and the
    # server should start instantly even when no model is installed.
    import ask
    from ingestion.documents import Document  # noqa: F401

    folder = Path(request.folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a folder: {folder}")

    candidates = sorted(
        p for p in folder.iterdir() if p.is_file() and _is_data_file(p)
    )[: request.max_files]

    if not candidates:
        raise ValueError(
            f"No indexable files in {folder}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    emit({"type": "progress", "stage": "scan",
          "message": f"Found {len(candidates)} file(s) in {folder.name}",
          "current": 0, "total": len(candidates)})

    documents = []
    for position, path in enumerate(candidates, 1):
        emit({"type": "progress", "stage": "extract",
              "message": f"Extracting {path.name}",
              "current": position, "total": len(candidates)})
        started = time.perf_counter()
        try:
            documents.extend(
                ask.load_file(path, ocr_lang=request.ocr_lang,
                              max_pages=request.max_pages)
            )
        except Exception as exc:
            # One unreadable file must not abandon an expensive multi-file run.
            emit({"type": "progress", "stage": "extract",
                  "message": f"Skipped {path.name}: {exc}",
                  "current": position, "total": len(candidates)})
            continue
        emit({"type": "progress", "stage": "extract",
              "message": f"{path.name}: {len(documents)} units "
                         f"({time.perf_counter() - started:.1f}s)",
              "current": position, "total": len(candidates)})

    if not documents:
        raise ValueError("No text could be extracted from any file.")

    emit({"type": "progress", "stage": "index",
          "message": f"Chunking and indexing {len(documents)} units",
          "current": len(candidates), "total": len(candidates)})

    collection = "folder_" + "".join(
        ch if ch.isalnum() else "_" for ch in folder.name
    )[:24] + f"_{abs(hash(str(folder))) % 10**8}"

    client, where = _qdrant_client()
    emit({"type": "progress", "stage": "index", "message": f"Vector store: {where}",
          "current": len(candidates), "total": len(candidates)})

    pipeline = ask.build_pipeline(
        documents,
        collection,
        backend=request.backend,
        model=request.model,
        qdrant_client=client,
    )

    key = _session_key(request)
    with _sessions_lock:
        _sessions[key] = pipeline

    emit({
        "type": "done",
        "session_id": key,
        "documents": len(documents),
        "files": len(candidates),
        "folder": str(folder),
        "backend": getattr(pipeline.llm, "name", request.backend),
        "model": getattr(pipeline.llm, "model", request.model or ""),
    })


@app.post("/api/index")
def index_folder(request: IndexRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_worker(lambda emit: _build(request, emit)),
        media_type="application/x-ndjson",
    )


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


def _answer(request: ChatRequest, emit) -> None:
    with _sessions_lock:
        pipeline = _sessions.get(request.session_id)
    if pipeline is None:
        raise ValueError("This folder is no longer indexed. Index it again.")

    question = request.question.strip()
    if not question:
        raise ValueError("Empty question.")

    emit({"type": "status", "message": "Retrieving evidence..."})

    result = pipeline.answer(
        question,
        stream_callback=lambda token: emit({"type": "token", "text": token}),
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
def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_worker(lambda emit: _answer(request, emit)),
        media_type="application/x-ndjson",
    )


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "sessions": len(_sessions)}


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run(host: str = "127.0.0.1", port: Optional[int] = None) -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the RAG chat server.")
    parser.add_argument("--host", type=str, default=host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=port or int(os.environ.get("PORT", "8080")), help="Port to bind to")
    args, _ = parser.parse_known_args()

    print(f"[*] Starting RAG Chat server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
