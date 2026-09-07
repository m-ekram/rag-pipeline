"""Phase 0 smoke check: embed a couple of documents and round-trip them through Qdrant.

Run with `python eval/check_qdrant.py` after `docker compose up -d`.
Exits non-zero if anything fails, so it is usable as a CI/preflight gate.
"""

import os
import sys
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "smoke_check"


def check_qdrant_and_embeddings():
    logger.info("Loading SentenceTransformer (%s)...", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    # Derive the vector size from the model instead of hardcoding 384, so swapping
    # the embedding model does not silently create a mismatched collection.
    vector_size = model.get_sentence_embedding_dimension()

    logger.info("Connecting to Qdrant at %s...", QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    docs = [
        "Qdrant is a vector database.",
        "Sentence transformers generate dense embeddings.",
    ]

    logger.info("Embedding %d documents (dim=%d)...", len(docs), vector_size)
    embeddings = model.encode(docs)

    logger.info("Upserting to Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=i, vector=embedding.tolist(), payload={"text": doc})
            for i, (doc, embedding) in enumerate(zip(docs, embeddings))
        ],
        wait=True,
    )

    logger.info("Searching Qdrant...")
    query = "What is Qdrant?"
    query_vector = model.encode(query).tolist()

    # `client.search()` was removed in qdrant-client 1.x; `query_points` replaces it
    # and returns a response object whose `.points` holds the ranked hits.
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1,
        with_payload=True,
    ).points

    if not hits:
        raise RuntimeError("Search returned no results — upsert or indexing failed.")

    top = hits[0]
    logger.info("Top hit: %s (score: %.4f)", top.payload["text"], top.score)

    if top.payload["text"] != docs[0]:
        raise RuntimeError(
            f"Unexpected top hit for {query!r}: {top.payload['text']!r}"
        )

    client.delete_collection(COLLECTION_NAME)
    logger.info("Success!")


if __name__ == "__main__":
    try:
        check_qdrant_and_embeddings()
    except Exception as exc:
        logger.error("Qdrant smoke check failed: %s", exc)
        sys.exit(1)
