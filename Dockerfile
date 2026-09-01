# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Both libraries look here for downloaded weights. Pointing them at one
    # directory lets compose mount a single cache volume, so bge-small is not
    # re-downloaded every time a container is recreated.
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .

# torch is not in requirements.txt; it arrives transitively via
# sentence-transformers. Installing it first from the CPU index is what stops
# pip resolving the default CUDA build, which drags in several GB of nvidia
# wheels this image can never use.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# llama-cpp-python publishes no wheel on PyPI, so a plain install triggers a
# source build that fails here - python:3.11-slim has no compiler. The extra
# index carries prebuilt CPU wheels.
RUN pip install --no-cache-dir \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      -r requirements.txt

COPY config.py ./
COPY app ./app
COPY eval ./eval
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 orbit \
    && mkdir -p /app/data /app/faiss_index /app/models /app/.embed_cache /app/.cache/huggingface \
    && chown -R orbit:orbit /app
USER orbit

# data and models are read-only mounts of host content; the index and the model
# cache are written at runtime. None are baked into the image: they are large,
# they change on a different cadence than the code, and they hold document text.
VOLUME ["/app/data", "/app/faiss_index", "/app/models", "/app/.embed_cache", "/app/.cache/huggingface"]

EXPOSE 8000

# Readiness, not liveness: /health answers 200 with status "not_ready" when the
# index is missing, so the status field is what actually has to be inspected.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import json,urllib.request,sys; \
sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8000/health'))['status']=='ok' else 1)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
