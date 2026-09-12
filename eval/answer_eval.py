"""End-to-end answer evaluation against a labelled question set.

    ragenv311\\Scripts\\python eval/answer_eval.py --backend ollama \
        --out docs/dissertation/data

Questions come from eval/labelled_questions.json. Each has the folder it is
asked of, the expected answer as one or more accepted strings (any one must
appear in the answer, case-insensitive, commas and spaces ignored in numbers),
the page(s) the fact is printed on, and whether the documents can answer it at
all. Unanswerable questions measure the system's willingness to decline.

Per question it records, through the real HTTP API exactly as the browser:
correctness, whether any cited page is an expected page, abstention, latency
(first token, total) and prompt size. The summary reports accuracy, answered
accuracy, citation precision, abstention rates on answerable and unanswerable
questions, and latency mean / sd / median / p90, with a Wilson 95% interval
on each proportion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.latency_probe import final, stream  # noqa: E402

QUESTIONS = Path(__file__).with_name("labelled_questions.json")
_PAGE = re.compile(r"Page (\d+)")


def _normal(text: str) -> str:
    """Lowercase; drop the thousands separators and spaces that vary in numbers."""
    return re.sub(r"(?<=\d)[,\s](?=\d)", "", text.lower())


def wilson(successes: int, n: int, z: float = 1.96) -> dict:
    if n == 0:
        return {"rate": None, "lo": None, "hi": None, "n": 0}
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half), "n": n}


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {"n": len(values), "mean": st.mean(values), "sd": st.stdev(values) if len(values) > 1 else 0.0,
            "median": st.median(values), "p90": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
            "min": ordered[0], "max": ordered[-1]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = []
    with httpx.Client(base_url=args.api) as client:
        sessions = {}
        for folder in sorted({q["folder"] for q in spec["questions"]}):
            path = (ROOT / spec["folders"][folder]).resolve()
            events, marks = stream(client, "/api/index", {"folder": str(path), "backend": args.backend,
                                                          "model": args.model, "ocr_lang": "auto"})
            done = final(events)
            if done.get("type") != "done":
                raise RuntimeError(f"indexing {folder} failed: {done.get('message')}")
            sessions[folder] = done["session_id"]
            print(f"[+] indexed {folder} in {marks['total_s']:.1f}s", flush=True)

        for item in spec["questions"]:
            events, marks = stream(client, "/api/chat", {"session_id": sessions[item["folder"]],
                                                         "question": item["question"]})
            result = final(events)
            answer = result.get("answer") or ""
            metrics = result.get("metrics") or {}
            abstained = bool(result.get("abstained"))
            cited_pages = {int(p) for p in _PAGE.findall(answer)}
            correct = (not abstained) and any(_normal(a) in _normal(answer) for a in item.get("accept", []))
            row = {
                "id": item["id"], "folder": item["folder"], "type": item["type"],
                "answerable": item["answerable"], "question": item["question"],
                "abstained": abstained, "correct": correct,
                "cited_expected_page": bool(cited_pages & set(item.get("pages", []))),
                "cited_pages": sorted(cited_pages), "expected_pages": item.get("pages", []),
                "first_token_s": marks.get("first_token_s"), "total_s": marks.get("total_s"),
                "input_tokens": metrics.get("input_tokens"), "output_tokens": metrics.get("output_tokens"),
                "retrieval_ms": metrics.get("latency_retrieval_ms"),
                "generation_ms": metrics.get("latency_generation_ms"),
                "backend": metrics.get("backend"), "model": metrics.get("model"),
                "answer": answer[:400],
            }
            rows.append(row)
            verdict = "ABSTAIN" if abstained else ("OK" if correct else "WRONG")
            print(f"  {verdict:7s} {row['total_s']:6.1f}s {item['id']}: {item['question']}", flush=True)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    summary = {
        "questions": len(rows), "answerable": len(answerable), "unanswerable": len(unanswerable),
        "accuracy": wilson(sum(r["correct"] for r in answerable), len(answerable)),
        "answered_accuracy": wilson(sum(r["correct"] for r in answered), len(answered)),
        "citation_page_precision": wilson(sum(r["cited_expected_page"] for r in answered if r["correct"]),
                                          sum(r["correct"] for r in answered)),
        "abstain_when_answerable": wilson(sum(r["abstained"] for r in answerable), len(answerable)),
        "abstain_when_unanswerable": wilson(sum(r["abstained"] for r in unanswerable), len(unanswerable)),
        "by_folder": {f: wilson(sum(r["correct"] for r in answerable if r["folder"] == f),
                                sum(1 for r in answerable if r["folder"] == f))
                      for f in sorted({r["folder"] for r in answerable})},
        "by_type": {t: wilson(sum(r["correct"] for r in answerable if r["type"] == t),
                              sum(1 for r in answerable if r["type"] == t))
                    for t in sorted({r["type"] for r in answerable})},
        "first_token_s": describe([r["first_token_s"] for r in rows if r["first_token_s"]]),
        "total_s": describe([r["total_s"] for r in rows if r["total_s"]]),
        "retrieval_ms": describe([r["retrieval_ms"] for r in rows if r["retrieval_ms"]]),
        "input_tokens": describe([r["input_tokens"] for r in rows if r["input_tokens"]]),
    }
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"answer_eval_{stamp}.json"
    target.write_text(json.dumps({"backend": args.backend, "model": args.model, "summary": summary,
                                  "questions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[+] wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
