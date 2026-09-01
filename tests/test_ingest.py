"""Ingestion entry point.

`clear_directory` exists because of a bug that only appears under Docker: with
the index directory mounted as a volume, `--rebuild` called shutil.rmtree on
the mount point itself and died with EBUSY. Clearing the contents works in both
places, so the test pins the property that matters - the directory survives.
"""

import shutil

import pytest

from app.ingest import clear_directory


def test_the_directory_itself_survives(tmp_path):
    """Removing the directory is what fails on a mounted volume."""
    (tmp_path / "index.faiss").write_text("x")
    before = tmp_path.stat().st_ino
    clear_directory(tmp_path)
    assert tmp_path.is_dir()
    assert tmp_path.stat().st_ino == before


def test_files_are_removed(tmp_path):
    for name in ("index.faiss", "index.pkl", "chunks.jsonl"):
        (tmp_path / name).write_text("x")
    clear_directory(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_nested_directories_are_removed(tmp_path):
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("x")
    clear_directory(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_already_empty_directory_is_fine(tmp_path):
    clear_directory(tmp_path)
    assert tmp_path.is_dir()


def test_it_does_not_recurse_above_itself(tmp_path):
    sibling = tmp_path / "keep.txt"
    target = tmp_path / "index"
    target.mkdir()
    sibling.write_text("x")
    (target / "gone.txt").write_text("x")
    clear_directory(target)
    assert sibling.exists()
    assert list(target.iterdir()) == []
