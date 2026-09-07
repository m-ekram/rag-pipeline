"""Offline tests for the Phase 0 dataset downloaders.

These deliberately avoid the network: the guard logic (dedup, retry caps,
partial-download detection, archive safety) is what has actually broken.
"""

import io
import json
import zipfile

import pytest

from eval.download_fiqa import _is_complete, _safe_extract
from eval import download_noisy_corpus


# --- download_fiqa -------------------------------------------------------


def test_is_complete_rejects_partial_extract(tmp_path):
    fiqa = tmp_path / "fiqa"
    (fiqa / "qrels").mkdir(parents=True)
    (fiqa / "corpus.jsonl").write_text("{}")
    assert not _is_complete(str(fiqa))

    (fiqa / "queries.jsonl").write_text("{}")
    (fiqa / "qrels" / "test.tsv").write_text("")
    assert _is_complete(str(fiqa))


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")

    dest = tmp_path / "dest"
    dest.mkdir()
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(RuntimeError, match="outside data dir"):
            _safe_extract(zf, str(dest))

    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_allows_normal_members(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("fiqa/corpus.jsonl", "{}")

    dest = tmp_path / "dest"
    dest.mkdir()
    with zipfile.ZipFile(archive) as zf:
        _safe_extract(zf, str(dest))

    assert (dest / "fiqa" / "corpus.jsonl").exists()


# --- download_noisy_corpus ----------------------------------------------


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _fake_page(page_id, title="T"):
    return {str(page_id): {"title": title, "extract": "x" * 500}}


def test_dedupes_repeated_pages_and_terminates(tmp_path, monkeypatch):
    """The random generator repeats pages; the corpus must not."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        # Always the same two pages — previously this appended duplicates
        # forever and never reached num_docs.
        pages = {**_fake_page(1, "A"), **_fake_page(2, "B")}
        return _FakeResponse(json.dumps({"query": {"pages": pages}}).encode())

    monkeypatch.setattr(download_noisy_corpus.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download_noisy_corpus.time, "sleep", lambda _s: None)
    monkeypatch.setattr(download_noisy_corpus, "__file__", str(tmp_path / "eval" / "m.py"))

    corpus_file = download_noisy_corpus.download_wikipedia_corpus(
        num_docs=10, max_batches=3
    )

    docs = json.loads(open(corpus_file, encoding="utf-8").read())
    ids = [d["id"] for d in docs]
    assert ids == ["1", "2"], "duplicate pages must collapse to unique ids"
    assert calls["n"] == 3, "loop must stop at max_batches, not spin forever"


def test_stops_after_consecutive_api_failures(tmp_path, monkeypatch):
    def always_fails(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(download_noisy_corpus.urllib.request, "urlopen", always_fails)
    monkeypatch.setattr(download_noisy_corpus.time, "sleep", lambda _s: None)
    monkeypatch.setattr(download_noisy_corpus, "__file__", str(tmp_path / "eval" / "m.py"))

    with pytest.raises(RuntimeError, match="failed"):
        download_noisy_corpus.download_wikipedia_corpus(num_docs=10, max_batches=100)


def test_short_extracts_are_skipped(tmp_path, monkeypatch):
    def fake_urlopen(req, timeout=None):
        pages = {
            "1": {"title": "Stub", "extract": "too short"},
            "2": {"title": "Real", "extract": "y" * 500},
        }
        return _FakeResponse(json.dumps({"query": {"pages": pages}}).encode())

    monkeypatch.setattr(download_noisy_corpus.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download_noisy_corpus.time, "sleep", lambda _s: None)
    monkeypatch.setattr(download_noisy_corpus, "__file__", str(tmp_path / "eval" / "m.py"))

    corpus_file = download_noisy_corpus.download_wikipedia_corpus(
        num_docs=5, max_batches=2
    )
    docs = json.loads(open(corpus_file, encoding="utf-8").read())
    assert [d["id"] for d in docs] == ["2"]


def test_existing_corpus_is_not_refetched(tmp_path, monkeypatch):
    def must_not_be_called(req, timeout=None):
        raise AssertionError("should not hit the network when corpus exists")

    monkeypatch.setattr(download_noisy_corpus.urllib.request, "urlopen", must_not_be_called)
    monkeypatch.setattr(download_noisy_corpus, "__file__", str(tmp_path / "eval" / "m.py"))

    target = tmp_path / "data" / "noisy_corpus"
    target.mkdir(parents=True)
    (target / "corpus.json").write_text("[]")

    assert download_noisy_corpus.download_wikipedia_corpus(num_docs=5) == str(
        target / "corpus.json"
    )
