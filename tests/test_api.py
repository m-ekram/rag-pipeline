"""The FastAPI surface, exercised against a stub engine.

TestClient is deliberately *not* used as a context manager: that skips the
lifespan hook, so no FAISS index and no 2GB GGUF are loaded and the whole file
runs in well under a second. The engine is injected into `api._state` instead.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import app.api as api

ANSWER = {
    "answer": "Use BackgroundTasks for work after the response [1].",
    "sources": [
        {"n": 1, "source": "fastapi/tutorial__background-tasks.md", "title": "Background Tasks",
         "page": None, "chunk_id": "abc123", "snippet": "Declare a BackgroundTasks parameter..."}
    ],
    "search_query": "How do I use BackgroundTasks?",
    "latency_ms": 42,
}


class StubEngine:
    meta = {"chunk_count": 3, "embedding_model": "BAAI/bge-small-en-v1.5"}

    def __init__(self):
        self.chunks = [
            Document(page_content="a", metadata={"source": "b.md"}),
            Document(page_content="b", metadata={"source": "a.md"}),
            Document(page_content="c", metadata={"source": "a.md"}),
        ]
        self.calls = []

    def ask(self, question, history=None, top_k=None):
        self.calls.append({"question": question, "history": history, "top_k": top_k})
        return dict(ANSWER)

    async def astream(self, question, history=None, top_k=None):
        yield {"type": "sources", "search_query": question, "sources": ANSWER["sources"]}
        yield {"type": "token", "text": "Use BackgroundTasks"}
        yield {"type": "done"}


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def engine():
    """Install a stub engine and restore whatever was there afterwards."""
    previous = dict(api._state)
    stub = StubEngine()
    api._state["engine"], api._state["error"] = stub, None
    yield stub
    api._state.clear()
    api._state.update(previous)


@pytest.fixture
def broken_engine():
    previous = dict(api._state)
    api._state["engine"], api._state["error"] = None, "No index at /app/faiss_index."
    yield
    api._state.clear()
    api._state.update(previous)


# -- health --------------------------------------------------------------------


def test_health_reports_ok_when_the_engine_loaded(client, engine):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["detail"] is None
    assert body["index"]["chunk_count"] == 3


def test_health_reports_not_ready_and_says_why(client, broken_engine):
    """Readiness must be answerable even when the engine is down."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_ready"
    assert "No index" in body["detail"]
    assert body["index"] == {}


def test_health_always_names_the_active_models(client, engine):
    body = client.get("/health").json()
    for key in ("provider", "chat_model", "embedding_model", "top_k", "hybrid_retrieval"):
        assert key in body


# -- sources -------------------------------------------------------------------


def test_sources_aggregates_chunk_counts_per_document(client, engine):
    body = client.get("/sources").json()
    assert body["document_count"] == 2
    assert body["chunk_count"] == 3
    assert body["documents"] == [{"source": "a.md", "chunks": 2}, {"source": "b.md", "chunks": 1}]


def test_sources_is_sorted_by_path(client, engine):
    sources = [d["source"] for d in client.get("/sources").json()["documents"]]
    assert sources == sorted(sources)


def test_sources_is_503_when_the_engine_is_down(client, broken_engine):
    assert client.get("/sources").status_code == 503


# -- ask -----------------------------------------------------------------------


def test_ask_returns_the_answer_with_citations(client, engine):
    body = client.post("/ask", json={"question": "How do I use BackgroundTasks?"}).json()
    assert body["answer"] == ANSWER["answer"]
    assert body["sources"][0]["source"].endswith("tutorial__background-tasks.md")
    assert body["search_query"] and isinstance(body["latency_ms"], int)


def test_ask_forwards_history_and_top_k(client, engine):
    client.post("/ask", json={
        "question": "and for Enterprise?",
        "history": [{"role": "user", "content": "Standard tier?"},
                    {"role": "assistant", "content": "600 rpm."}],
        "top_k": 3,
    })
    call = engine.calls[-1]
    assert call["top_k"] == 3
    assert [t["role"] for t in call["history"]] == ["user", "assistant"]


def test_ask_defaults_history_to_empty_and_top_k_to_none(client, engine):
    client.post("/ask", json={"question": "hello there"})
    assert engine.calls[-1] == {"question": "hello there", "history": [], "top_k": None}


def test_ask_is_503_when_the_engine_is_down(client, broken_engine):
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 503
    assert "No index" in response.json()["detail"]


@pytest.mark.parametrize("payload", [
    {},                                             # question is required
    {"question": ""},                               # min_length
    {"question": "x" * 2001},                       # max_length
    {"question": "ok", "top_k": 0},                 # ge=1
    {"question": "ok", "top_k": 21},                # le=20
    {"question": "ok", "history": [{"role": "system", "content": "x"}]},  # role pattern
    {"question": "ok", "history": [{"role": "user"}]},                    # content required
])
def test_malformed_requests_are_rejected_with_422(client, engine, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_a_rejected_request_never_reaches_the_engine(client, engine):
    client.post("/ask", json={"question": ""})
    assert engine.calls == []


# -- streaming -----------------------------------------------------------------


def test_ask_stream_emits_server_sent_events(client, engine):
    response = client.post("/ask/stream", json={"question": "How do I use BackgroundTasks?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = [json.loads(line[len("data: "):])
              for line in response.text.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["sources", "token", "done"]
    assert events[0]["sources"][0]["n"] == 1


def test_ask_stream_disables_proxy_buffering(client, engine):
    """Without this header nginx holds the whole stream until it completes."""
    response = client.post("/ask/stream", json={"question": "hello there"})
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"


def test_ask_stream_is_503_when_the_engine_is_down(client, broken_engine):
    assert client.post("/ask/stream", json={"question": "anything"}).status_code == 503


# -- reload --------------------------------------------------------------------


def test_reload_reopens_the_index(client, engine, monkeypatch):
    calls = []
    monkeypatch.setattr(api, "_load_engine", lambda: calls.append(1))
    body = client.post("/reload").json()
    assert calls == [1]
    assert body["status"] == "ok"


def test_reload_reports_failure_without_raising(client, monkeypatch):
    previous = dict(api._state)
    try:
        def fail():
            api._state["engine"], api._state["error"] = None, "still no index"

        monkeypatch.setattr(api, "_load_engine", fail)
        body = client.post("/reload").json()
        assert body["status"] == "not_ready"
        assert body["detail"] == "still no index"
    finally:
        api._state.clear()
        api._state.update(previous)


# -- ui and docs ---------------------------------------------------------------


def test_the_root_path_serves_the_chat_ui(client, engine):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_openapi_is_available(client, engine):
    schema = client.get("/openapi.json").json()
    assert "/ask" in schema["paths"]
    assert "/health" in schema["paths"]


def test_the_request_schema_is_published(client, engine):
    """Requests are modelled; responses are not yet (see M4)."""
    schema = client.get("/openapi.json").json()
    assert "AskRequest" in schema["components"]["schemas"]
