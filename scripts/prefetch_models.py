"""Download every model the pipeline loads, once, before the first request.

Without this, the first "Index folder" in the UI silently downloads about 1 GB
of weights inside the request, and the user sees minutes of nothing.

    python scripts/prefetch_models.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rerank.cross_encoder import DEFAULT_MODEL, MULTILINGUAL_LIGHT  # noqa: E402
from retrieval.embedder import MULTILINGUAL_MODEL  # noqa: E402

EMBEDDERS = [MULTILINGUAL_MODEL]
RERANKERS = [DEFAULT_MODEL, MULTILINGUAL_LIGHT]


def main() -> int:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    for name in EMBEDDERS:
        started = time.perf_counter()
        print(f"[*] Embedder {name} ...", flush=True)
        SentenceTransformer(name)
        print(f"[+] ready in {time.perf_counter() - started:.1f}s", flush=True)
    for name in RERANKERS:
        started = time.perf_counter()
        print(f"[*] Reranker {name} ...", flush=True)
        CrossEncoder(name)
        print(f"[+] ready in {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
