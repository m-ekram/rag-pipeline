"""The API's collection naming, heartbeats and cancellation."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
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


# --- streaming ------------------------------------------------------------


class _Request:
    async def is_disconnected(self):
        return False


async def _collect(stream):
    return [json.loads(line) async for line in stream]


def test_stream_sends_heartbeats_while_the_work_is_silent(monkeypatch):
    """A long OCR or model load used to leave the stream silent, and the UI
    (and the proxy's timeout) could not tell busy from dead."""
    from api import server

    monkeypatch.setattr(server, "HEARTBEAT_SECONDS", 0.05)

    def work(emit):
        time.sleep(0.4)
        emit({"type": "done"})

    events = asyncio.run(_collect(server._stream_worker(_Request(), work)))
    kinds = [e["type"] for e in events]
    assert "heartbeat" in kinds
    assert kinds[-1] == "done"


def test_disconnect_stops_the_running_work(monkeypatch):
    """An abandoned answer used to keep generating on the single pipeline
    thread, and every later question queued silently behind it."""
    from api import server

    monkeypatch.setattr(server, "HEARTBEAT_SECONDS", 0.05)
    produced, finished = [], threading.Event()

    def work(emit):
        try:
            for i in range(500):
                produced.append(i)
                emit({"type": "token", "text": str(i)})
                time.sleep(0.01)
        finally:
            finished.set()

    async def read_three_then_leave():
        stream = server._stream_worker(_Request(), work)
        seen = 0
        async for _line in stream:
            seen += 1
            if seen == 3:
                break
        await stream.aclose()  # what the server does when the client goes away

    asyncio.run(read_three_then_leave())
    assert finished.wait(5)
    assert len(produced) < 500
