# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/app/.cache/models \
    EMBED_CACHE_DIR=/app/.cache/embed

WORKDIR /app

# Dependencies first so code edits do not bust the layer cache. No PyTorch:
# local embeddings and re-ranking run on ONNX Runtime.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py ./
COPY app ./app
COPY eval ./eval
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /app/.cache/models /app/.cache/embed /app/faiss_index \
    && chown -R rag:rag /app
USER rag

# The index is a mounted volume, not a baked layer: it is large, it changes on a
# different cadence than the code, and it holds document content. The cache
# volume keeps downloaded models and paid-for embeddings across restarts.
VOLUME ["/app/data", "/app/faiss_index", "/app/.cache"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
