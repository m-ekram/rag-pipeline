"""FastAPI service wrapping the RAG engine.

    uvicorn app.api:app --reload

    GET  /              chat UI
    GET  /health        readiness + index metadata
    GET  /sources       documents currently indexed
    POST /ask           JSON answer with citations
    POST /ask/stream    same, streamed as server-sent events
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from app.rag import RagEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UI_FILE = Path(__file__).parent / "ui" / "index.html"

_state: dict = {"engine": None, "error": None}


def _load_engine() -> None:
    try:
        _state["engine"] = RagEngine()
        _state["error"] = None
    except SystemExit as exc:  # missing index / key -> serve a useful message
        _state["engine"], _state["error"] = None, str(exc)
        logger.error("Engine not ready: %s", exc)
    except Exception as exc:
        _state["engine"], _state["error"] = None, f"{exc.__class__.__name__}: {exc}"
        logger.exception("Engine failed to load")


def _engine() -> RagEngine:
    if _state["engine"] is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "Engine not ready.")
    return _state["engine"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_engine()
    yield


app = FastAPI(title="RAG Document Q&A", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$", description="user or assistant")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(
        default_factory=list,
        description="Prior turns. The API is stateless; history is passed per request.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Passages to retrieve. Defaults to TOP_K."
    )


class Source(BaseModel):
    """One retrieved passage, numbered to match the [n] markers in the answer."""

    n: int = Field(description="Citation number referenced in the answer text")
    source: str = Field(description="Path relative to the corpus root")
    title: str | None = None
    page: int | None = Field(default=None, description="1-based page number, PDFs only")
    chunk_id: str | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    search_query: str = Field(
        description="The question after history-aware rewriting; equals the "
        "question when there is no history."
    )
    latency_ms: int


class IndexMeta(BaseModel):
    """Metadata written at ingest time, empty when no index is loaded."""

    embed_provider: str | None = None
    chat_provider: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    chunk_count: int | None = None
    built_at: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    detail: str | None = Field(default=None, description="Why the engine is not ready")
    provider: str
    chat_model: str
    embedding_model: str
    top_k: int
    hybrid_retrieval: bool
    index: IndexMeta


class IndexedDocument(BaseModel):
    source: str
    chunks: int


class SourcesResponse(BaseModel):
    documents: list[IndexedDocument]
    document_count: int
    chunk_count: int


class ReloadResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    detail: str | None = None


@app.get("/health", response_model=HealthResponse, summary="Readiness and active configuration")
def health() -> dict:
    engine = _state["engine"]
    return {
        "status": "ok" if engine else "not_ready",
        "detail": _state["error"],
        "provider": config.PROVIDER,
        "chat_model": config.CHAT_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "top_k": config.TOP_K,
        "hybrid_retrieval": config.USE_HYBRID,
        "index": engine.meta if engine else {},
    }


@app.get("/sources", response_model=SourcesResponse, summary="Documents currently indexed")
def sources() -> dict:
    engine = _engine()
    counts: dict[str, int] = {}
    for chunk in engine.chunks:
        key = chunk.metadata.get("source", "unknown")
        counts[key] = counts.get(key, 0) + 1
    documents = [{"source": k, "chunks": v} for k, v in sorted(counts.items())]
    return {"documents": documents, "document_count": len(documents), "chunk_count": len(engine.chunks)}


@app.post("/ask", response_model=AskResponse, summary="Answer a question with citations")
async def ask(req: AskRequest) -> dict:
    engine = _engine()
    history = [t.model_dump() for t in req.history]
    # The chain is sync and network-bound; keep the event loop free.
    return await asyncio.to_thread(engine.ask, req.question, history, req.top_k)


@app.post(
    "/ask/stream",
    summary="Same as /ask, streamed as server-sent events",
    response_description=(
        "text/event-stream of JSON events: {type: sources|token|replace|done|error}"
    ),
)
async def ask_stream(req: AskRequest) -> StreamingResponse:
    engine = _engine()
    history = [t.model_dump() for t in req.history]

    async def events():
        async for event in engine.astream(req.question, history, req.top_k):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reload", response_model=ReloadResponse, summary="Re-open the index after a re-ingest")
def reload_index() -> dict:
    """Re-open the index after a re-ingest, without restarting the process."""
    _load_engine()
    return {"status": "ok" if _state["engine"] else "not_ready", "detail": _state["error"]}


@app.get("/")
def index_page():
    if not UI_FILE.exists():
        return {"message": "UI not found. API is at /docs"}
    return FileResponse(UI_FILE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host=config.HOST, port=config.PORT, reload=True)
