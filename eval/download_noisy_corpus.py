"""Download a corpus of random Wikipedia articles to act as retrieval noise."""

import os
import json
import time
import logging
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&generator=random&grnnamespace=0&grnlimit=10"
    "&prop=extracts&explaintext=1&format=json"
)
MIN_TEXT_CHARS = 200
MAX_CONSECUTIVE_ERRORS = 5


def download_wikipedia_corpus(num_docs=150, max_batches=100):
    """Fetch `num_docs` random Wikipedia articles into data/noisy_corpus/corpus.json.

    `max_batches` bounds the loop: the random generator returns duplicates and
    short stubs, so without a cap a persistently failing API would spin forever.
    """
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "noisy_corpus"
    )
    os.makedirs(data_dir, exist_ok=True)

    corpus_file = os.path.join(data_dir, "corpus.json")
    if os.path.exists(corpus_file):
        logger.info("Noisy corpus already exists at %s", corpus_file)
        return corpus_file

    logger.info("Downloading %d random Wikipedia articles as noisy corpus...", num_docs)

    # Keyed by page id: the random generator can hand back the same page in
    # different batches, and duplicates would corrupt the retrieval metrics.
    documents = {}
    consecutive_errors = 0

    for _ in range(max_batches):
        if len(documents) >= num_docs:
            break

        try:
            req = urllib.request.Request(
                API_URL, headers={"User-Agent": "RAG-Eval/0.1 (noisy corpus builder)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            consecutive_errors = 0
        except (urllib.error.URLError, OSError, ValueError) as exc:
            consecutive_errors += 1
            logger.error("Error fetching data (%d/%d): %s",
                         consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(
                    f"Wikipedia API failed {consecutive_errors} times in a row."
                ) from exc
            time.sleep(2 * consecutive_errors)
            continue

        for page_id, page_data in data.get("query", {}).get("pages", {}).items():
            text = page_data.get("extract", "")
            if len(text.strip()) > MIN_TEXT_CHARS:
                documents[str(page_id)] = {
                    "id": str(page_id),
                    "title": page_data.get("title", ""),
                    "text": text,
                }

        logger.info("Fetched %d/%d documents...", len(documents), num_docs)
        time.sleep(1)  # Be nice to the API

    if len(documents) < num_docs:
        logger.warning(
            "Only collected %d/%d documents after %d batches.",
            len(documents), num_docs, max_batches,
        )

    corpus = list(documents.values())[:num_docs]

    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d documents to %s", len(corpus), corpus_file)
    return corpus_file


if __name__ == "__main__":
    download_wikipedia_corpus(150)
