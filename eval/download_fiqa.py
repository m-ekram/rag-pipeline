"""Download the BEIR FiQA dataset used as retrieval ground truth."""

import os
import shutil
import zipfile
import logging
import tempfile
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
# A run is only complete if all of these landed; a partial extract must not be
# mistaken for a finished download on the next run.
EXPECTED_FILES = ("corpus.jsonl", "queries.jsonl", os.path.join("qrels", "test.tsv"))


def _is_complete(extract_path):
    return all(
        os.path.exists(os.path.join(extract_path, name)) for name in EXPECTED_FILES
    )


def _safe_extract(zip_ref, dest):
    """Extract while rejecting members that escape `dest` (zip-slip)."""
    dest = os.path.realpath(dest)
    for member in zip_ref.namelist():
        target = os.path.realpath(os.path.join(dest, member))
        if not (target == dest or target.startswith(dest + os.sep)):
            raise RuntimeError(f"Refusing to extract member outside data dir: {member}")
    zip_ref.extractall(dest)


def download_fiqa():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    extract_path = os.path.join(data_dir, "fiqa")

    if _is_complete(extract_path):
        logger.info("FiQA already exists at %s", extract_path)
        return extract_path

    if os.path.exists(extract_path):
        logger.warning("Removing incomplete FiQA download at %s", extract_path)
        shutil.rmtree(extract_path)

    logger.info("Downloading FiQA from %s...", URL)
    # Download to a temp file so an interrupted run leaves nothing behind.
    fd, tmp_zip = tempfile.mkstemp(suffix=".zip", dir=data_dir)
    os.close(fd)
    try:
        urllib.request.urlretrieve(URL, tmp_zip)

        logger.info("Extracting FiQA...")
        with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
            _safe_extract(zip_ref, data_dir)
    except BaseException:
        # Includes KeyboardInterrupt — a half-extracted dir would be silently
        # accepted by the guard above on the next run.
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path, ignore_errors=True)
        raise
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

    if not _is_complete(extract_path):
        raise RuntimeError(f"FiQA archive did not contain {EXPECTED_FILES}")

    logger.info("Done extracting to %s", extract_path)
    return extract_path


if __name__ == "__main__":
    download_fiqa()
