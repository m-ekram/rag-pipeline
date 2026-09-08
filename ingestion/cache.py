"""Content-addressed cache for page extraction (native text and OCR).

OCR is the most expensive stage in this project by a wide margin: the electoral
rolls are pure scans, so every page costs ~6s of PaddleOCR. Re-ingesting the
same PDF — after a chunking change, an embedding change, or simply a restart —
would otherwise re-pay that cost in full.

The key is derived from the file's *content hash*, not its path or mtime, so a
renamed or moved PDF still hits the cache, and an edited one correctly misses.
Every parameter that changes the output (render scale, OCR engine, recognition
width, thresholds) is folded into the key: a cache hit therefore guarantees the
text is exactly what a fresh run would produce.
"""

import os
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "extraction"
)
CACHE_VERSION = "4"  # bumped to invalidate extractions for multi-tier year header normalization


def file_digest(path: str, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed so large PDFs stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def summary(self) -> str:
        return (f"cache: {self.hits} hits / {self.misses} misses "
                f"({self.hit_rate:.0%} hit rate), {self.writes} written")


class ExtractionCache:
    """Disk cache of per-page extraction results.

    Entries are plain JSON so they can be inspected, diffed and version-
    controlled if wanted; the OCR text of a whole electoral roll is a few
    hundred KB, far cheaper than re-running OCR.
    """

    def __init__(self, root: str = DEFAULT_CACHE_DIR, *, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        self.stats = CacheStats()
        self._digests: dict[str, str] = {}  # path -> content hash, memoised

    # -- keys ------------------------------------------------------------

    def document_digest(self, path: str) -> str:
        """Content hash of a source file, computed at most once per process."""
        key = os.path.abspath(path)
        if key not in self._digests:
            self._digests[key] = file_digest(path)
        return self._digests[key]

    def page_key(self, path: str, page: int, params: dict[str, Any]) -> str:
        """Stable key for one page under one exact set of extraction settings."""
        payload = json.dumps(params, sort_keys=True, ensure_ascii=True, default=str)
        raw = f"{CACHE_VERSION}|{self.document_digest(path)}|{page}|{payload}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> str:
        # Shard by the first two hex chars: a 5,000-page corpus in one flat
        # directory makes listing and filesystem lookups slow.
        return os.path.join(self.root, key[:2], f"{key}.json")

    # -- access ----------------------------------------------------------

    def get(self, key: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        path = self._path_for(key)
        try:
            with open(path, encoding="utf-8") as handle:
                entry = json.load(handle)
        except FileNotFoundError:
            self.stats.misses += 1
            return None
        except (OSError, json.JSONDecodeError) as exc:
            # A truncated entry (killed mid-write) must degrade to a miss, not
            # take down an hours-long ingest.
            logger.warning("Discarding unreadable cache entry %s: %s", key[:12], exc)
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        return entry.get("value")

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        # Never store an empty extraction. A genuinely blank page is cheap to
        # redo, but an empty result produced by a broken OCR engine would be
        # served from cache forever: the transient failure becomes permanent,
        # and re-running after the fix silently returns nothing. Guarding here
        # rather than at each call site covers every writer.
        if isinstance(value, dict) and not str(value.get("text", "")).strip():
            logger.debug("Refusing to cache empty extraction for %s", key[:12])
            return
        path = self._path_for(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry = {"version": CACHE_VERSION, "written_at": time.time(), "value": value}
        # Write-then-rename so a crash cannot leave a half-written entry that a
        # later run would read as valid.
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(entry, handle, ensure_ascii=False)
            os.replace(tmp, path)
            self.stats.writes += 1
        except OSError as exc:
            logger.warning("Could not write cache entry %s: %s", key[:12], exc)
            if os.path.exists(tmp):
                os.remove(tmp)

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed."""
        removed = 0
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for name in filenames:
                if name.endswith(".json"):
                    os.remove(os.path.join(dirpath, name))
                    removed += 1
        return removed
