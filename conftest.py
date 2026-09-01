"""Put the repo root on sys.path so `config`, `app.*` and `eval.*` import
the same way under pytest as they do under `python -m app.ingest`."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
