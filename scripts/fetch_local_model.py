"""Download the local chat model (GGUF) into models/.

    python scripts/fetch_local_model.py
    python scripts/fetch_local_model.py --model qwen1.5b

Quantised GGUF weights run on CPU through llama.cpp. Q4_K_M is the usual
sweet spot: roughly a quarter the size of fp16 with little quality loss, which
is what makes a 3B model practical on a laptop CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS = {
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf", "~2.0 GB, best citation discipline at this size"),
    "qwen1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", "~1.0 GB, ~2x faster, weaker instruction following"),
    "llama3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "~2.0 GB, comparable to qwen3b"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a local GGUF chat model.")
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen3b")
    parser.add_argument("--out", default="models")
    args = parser.parse_args(argv)

    repo, filename, note = MODELS[args.model]
    dest = Path(args.out) / filename
    if dest.is_file():
        print(f"Already present: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return 0

    print(f"Downloading {args.model} ({note})\n  {repo}/{filename}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=args.out)
    size = Path(path).stat().st_size / 1e9
    print(f"\nSaved {path} ({size:.2f} GB)")
    print(f"\nPoint .env at it:\n    CHAT_MODEL={args.out}/{filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
