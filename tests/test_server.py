"""The API's per-folder vector collection must survive a server restart."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _name_in_fresh_process(folder: Path, hash_seed: str) -> str:
    code = (
        "import sys; from pathlib import Path; "
        "from api.server import _collection_name; "
        "print(_collection_name(Path(sys.argv[1])))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(folder)],
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_collection_name_is_stable_and_path_derived(tmp_path):
    from api.server import _collection_name

    folder = tmp_path / "Electoral Rolls"
    name = _collection_name(folder)
    assert name == _collection_name(folder)
    assert name.startswith("folder_Electoral_Rolls_")
    assert _collection_name(tmp_path / "other") != name


def test_collection_name_survives_a_restart(tmp_path):
    """`hash()` is salted per process, so the old name changed on every restart:
    each restart re-embedded the whole folder and orphaned the old collection."""
    folder = tmp_path / "rolls"
    assert _name_in_fresh_process(folder, "1") == _name_in_fresh_process(folder, "2")
