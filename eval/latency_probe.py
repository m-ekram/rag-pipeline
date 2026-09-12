"""Drive the running API end to end and time what a user actually waits for.

    python eval/latency_probe.py --folder data/demo/pmp --folder data/demo/rolls \
        --folder data/demo/paper --backend auto --out docs/dissertation/data

For each folder it POSTs /api/index, then asks that folder's golden questions
(eval/golden_questions.json, keyed by folder name) through POST /api/chat,
exactly as the browser does. Measured client-side: time to first byte, to the
first status line, to the first answer token, and in total. Measured
server-side: the per-stage `latency_*_ms` from the final event. Results go to
<out>/latency_<timestamp>.json and .csv, so every number reported is traceable
to one run of this script.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = Path(__file__).with_name("golden_questions.json")


def stream(client: httpx.Client, url: str, body: dict) -> tuple[list[dict], dict]:
    """POST and read the NDJSON stream, stamping when each kind of event first arrived."""
    started = time.perf_counter()
    marks: dict[str, float] = {}
    events: list[dict] = []
    with client.stream("POST", url, json=body, timeout=None) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            now = round(time.perf_counter() - started, 3)
            marks.setdefault("first_byte_s", now)
            if not line.strip():
                continue
            event = json.loads(line)
            marks.setdefault(f"first_{event.get('type')}_s", now)
            events.append(event)
    marks["total_s"] = round(time.perf_counter() - started, 3)
    return events, marks


def final(events: list[dict]) -> dict:
    for event in reversed(events):
        if event.get("type") in ("done", "error"):
            return event
    return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--folder", action="append", required=True, help="folder to index (repeatable)")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-pages", type=int, default=None, help="cap pages per PDF (quick runs)")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    rows: list[dict] = []
    runs: list[dict] = []

    with httpx.Client(base_url=args.api) as client:
        health = client.get("/api/health").json()
        print(f"[*] API {args.api} — warm-up: {health.get('warm')}")

        for folder in args.folder:
            folder_path = Path(folder).resolve()
            key = folder_path.name
            print(f"\n[*] Indexing {folder_path} ...", flush=True)
            events, marks = stream(client, "/api/index", {
                "folder": str(folder_path), "backend": args.backend, "model": args.model,
                "ocr_lang": "auto", "max_pages": args.max_pages,
            })
            done = final(events)
            index_run = {"folder": key, **marks, "result": done.get("type"),
                         "documents": done.get("documents"), "backend": done.get("backend"),
                         "model": done.get("model"), "error": done.get("message")}
            runs.append({"index": index_run, "log": [e.get("message") for e in events if e.get("message")]})
            print(f"    {done.get('type')}: {done.get('documents')} units in {marks['total_s']:.1f}s "
                  f"({done.get('backend')} · {done.get('model')}) {done.get('message') or ''}")
            if done.get("type") != "done":
                continue

            for question in questions.get(key, []):
                events, marks = stream(client, "/api/chat", {"session_id": done["session_id"], "question": question})
                answer = final(events)
                metrics = answer.get("metrics") or {}
                row = {
                    "folder": key,
                    "question": question,
                    "result": answer.get("type"),
                    "decision": answer.get("decision"),
                    "backend": metrics.get("backend"),
                    "model": metrics.get("model"),
                    "evidence_used": metrics.get("n_evidence_used"),
                    "input_tokens": metrics.get("input_tokens"),
                    "output_tokens": metrics.get("output_tokens"),
                    "first_byte_s": marks.get("first_byte_s"),
                    "first_status_s": marks.get("first_status_s"),
                    "first_token_s": marks.get("first_token_s"),
                    "total_s": marks.get("total_s"),
                    "retrieval_ms": metrics.get("latency_retrieval_ms"),
                    "rerank_ms": metrics.get("latency_rerank_ms"),
                    "generation_ms": metrics.get("latency_generation_ms"),
                    "answer": (answer.get("answer") or answer.get("message") or "")[:300],
                }
                rows.append(row)
                print(f"    {row['total_s']:6.2f}s  first token {row['first_token_s']}  "
                      f"[{row['decision']}] {question}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"latency_{stamp}.json").write_text(
        json.dumps({"api": args.api, "backend": args.backend, "model": args.model,
                    "max_pages": args.max_pages, "health": health, "index_runs": runs,
                    "questions": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        with open(out / f"latency_{stamp}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n[+] Wrote {out / f'latency_{stamp}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
