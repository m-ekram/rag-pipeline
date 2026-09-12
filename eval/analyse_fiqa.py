"""Statistics for the FiQA benchmark written by eval/benchmark_fiqa.py.

    ragenv311\\Scripts\\python eval/analyse_fiqa.py docs/dissertation/data/fiqa_benchmark.json

Computes, per contamination level:

- mean retrieval metrics per variant with bootstrap 95% confidence intervals;
- paired randomization tests on per-query nDCG@10 between variants;
- selective retrieval: risk-coverage curves and AURC for each confidence
  signal, where "risk" is the share of answered queries whose top five
  documents hold no relevant one (1 - Hit@5) under the deployed pipeline
  (hybrid + rerank);
- abstention thresholds calibrated on the dev queries at a target coverage and
  reported on the untouched test queries, plus the realistic drift case: the
  threshold fitted at rho = 0 applied unchanged at higher contamination;
- per-query latency statistics.

Writes <input>_summary.json and prints Markdown tables. Deterministic (seeded).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

METRICS = ("ndcg10", "r10", "r100", "mrr10", "p5", "hit5")
VARIANTS = ("bm25", "dense", "hybrid", "hybrid_rerank")
PAIRS = (("hybrid_rerank", "hybrid"), ("hybrid", "dense"), ("hybrid", "bm25"), ("dense", "bm25"))
SIGNALS = ("rerank", "rrf", "dense", "bm25")
TARGET_COVERAGES = (0.8, 0.6)
SEED = 13


def bootstrap(values: np.ndarray, n: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5))}


def randomization_test(a: np.ndarray, b: np.ndarray, n: int = 10000) -> dict:
    """Two-sided paired sign-flip test on the mean per-query difference."""
    diff = a - b
    rng = np.random.default_rng(SEED)
    null = (rng.choice([-1.0, 1.0], size=(n, len(diff))) * diff).mean(axis=1)
    observed = diff.mean()
    p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (n + 1)
    return {"diff": float(observed), "p": float(p),
            "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum())}


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> dict:
    order = np.argsort(-confidence, kind="stable")
    kept = correct[order]
    k = np.arange(1, len(kept) + 1)
    risk = 1 - np.cumsum(kept) / k
    oracle = 1 - np.cumsum(np.sort(correct)[::-1]) / k
    aurc = float(risk.mean())
    return {
        "aurc": aurc,
        "e_aurc": aurc - float(oracle.mean()),  # excess over a perfect ranking of queries
        "risk_at": {f"{c:.1f}": float(risk[max(1, math.ceil(c * len(kept))) - 1])
                    for c in (1.0, 0.8, 0.6, 0.4)},
        "curve": {"coverage": [float(x) for x in (k / len(kept))[::max(1, len(kept) // 60)]],
                  "risk": [float(x) for x in risk[::max(1, len(kept) // 60)]]},
    }


def threshold_for_coverage(confidence: np.ndarray, coverage: float) -> float:
    """Lowest confidence still answered when the top `coverage` share is kept."""
    ordered = np.sort(confidence)[::-1]
    return float(ordered[max(1, math.ceil(coverage * len(ordered))) - 1])


def apply_threshold(confidence: np.ndarray, correct: np.ndarray, tau: float) -> dict:
    answered = confidence >= tau
    coverage = float(answered.mean())
    risk = float(1 - correct[answered].mean()) if answered.any() else 0.0
    return {"coverage": coverage, "risk": risk}


def summarise_level(level: dict) -> dict:
    test = [r for r in level["queries"] if r["split"] == "test"]
    dev = [r for r in level["queries"] if r["split"] == "dev"]
    col = lambda rows, v, m: np.array([r["m"][v][m] for r in rows])  # noqa: E731

    variants = {v: {m: bootstrap(col(test, v, m)) for m in METRICS} for v in VARIANTS}
    significance = [{"a": a, "b": b, "metric": "ndcg10",
                     **randomization_test(col(test, a, "ndcg10"), col(test, b, "ndcg10"))}
                    for a, b in PAIRS]

    correct_test = col(test, "hybrid_rerank", "hit5")
    correct_dev = col(dev, "hybrid_rerank", "hit5")
    selective = {}
    for signal in SIGNALS:
        conf_test = np.array([r["conf"][signal] for r in test])
        selective[signal] = risk_coverage(conf_test, correct_test)

    conf_test = np.array([r["conf"]["rerank"] for r in test])
    conf_dev = np.array([r["conf"]["rerank"] for r in dev])
    calibration = {}
    for target in TARGET_COVERAGES:
        tau = threshold_for_coverage(conf_dev, target) if len(conf_dev) else float("nan")
        calibration[f"{target:.1f}"] = {
            "tau": tau,
            "dev": apply_threshold(conf_dev, correct_dev, tau) if len(conf_dev) else None,
            "test": apply_threshold(conf_test, correct_test, tau),
        }

    ms = lambda key, rows: np.array([r["ms"][key] for r in rows])  # noqa: E731
    full_rerank = [r for r in test if r["ms"]["rerank_pairs"] >= 15]
    latency = {
        "bm25_ms": {"mean": float(ms("bm25", test).mean()), "p95": float(np.percentile(ms("bm25", test), 95))},
        "dense_search_ms": {"mean": float(ms("dense_search", test).mean()),
                            "p95": float(np.percentile(ms("dense_search", test), 95))},
        "rerank15_ms": ({"mean": float(ms("rerank", full_rerank).mean()),
                         "p95": float(np.percentile(ms("rerank", full_rerank), 95)),
                         "n": len(full_rerank)} if full_rerank else None),
    }
    return {
        "rho": level["rho"], "docs": level["docs"], "chunks": level["chunks"],
        "test_queries": len(test), "dev_queries": len(dev),
        "variants": variants, "significance": significance,
        "ungated_risk": float(1 - correct_test.mean()),
        "selective": selective, "calibration": calibration, "latency": latency,
        "_conf_test": conf_test, "_correct_test": correct_test,
    }


def main(argv=None) -> int:
    path = Path((argv or sys.argv[1:] or ["docs/dissertation/data/fiqa_benchmark.json"])[0])
    data = json.loads(path.read_text(encoding="utf-8"))
    levels = [summarise_level(level) for level in data["levels"]]

    # Deployment drift: thresholds fitted on the cleanest corpus, applied unchanged.
    base = levels[0]["calibration"]
    drift = []
    for level in levels:
        for target, cal in base.items():
            drift.append({"rho": level["rho"], "target_coverage": float(target), "tau_rho0": cal["tau"],
                          "fixed": apply_threshold(level["_conf_test"], level["_correct_test"], cal["tau"]),
                          "recalibrated": level["calibration"][target]["test"]})
    for level in levels:
        level.pop("_conf_test")
        level.pop("_correct_test")

    summary = {"source": path.name, "config": data["config"], "query_embed_ms": data["query_embed_ms"],
               "embed_seconds": data.get("embed_seconds"), "levels": levels, "drift": drift}
    out = path.with_name(path.stem + "_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for level in levels:
        print(f"\n### rho = {level['rho']}  ({level['docs']} docs, {level['chunks']} chunks, "
              f"{level['test_queries']} test queries)\n")
        print("| Variant | nDCG@10 [95% CI] | Recall@10 | Recall@100 | MRR@10 | P@5 | Hit@5 |")
        print("|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            m = level["variants"][v]
            print(f"| {v} | {m['ndcg10']['mean']:.3f} [{m['ndcg10']['lo']:.3f}, {m['ndcg10']['hi']:.3f}] "
                  f"| {m['r10']['mean']:.3f} | {m['r100']['mean']:.3f} | {m['mrr10']['mean']:.3f} "
                  f"| {m['p5']['mean']:.3f} | {m['hit5']['mean']:.3f} |")
        print()
        for s in level["significance"]:
            print(f"- {s['a']} vs {s['b']}: dNDCG@10 {s['diff']:+.4f}, p = {s['p']:.4f} "
                  f"({s['wins']} wins / {s['losses']} losses)")
        print(f"\nungated risk {level['ungated_risk']:.3f}; AURC " +
              ", ".join(f"{k} {v['aurc']:.3f}" for k, v in level["selective"].items()))
        for target, cal in level["calibration"].items():
            print(f"- target coverage {target}: tau {cal['tau']:.3f} -> test coverage "
                  f"{cal['test']['coverage']:.3f}, risk {cal['test']['risk']:.3f}")
    print("\n### drift (rho0 thresholds applied unchanged)")
    for d in drift:
        print(f"- rho {d['rho']} target {d['target_coverage']}: fixed cov {d['fixed']['coverage']:.3f} "
              f"risk {d['fixed']['risk']:.3f} | recalibrated cov {d['recalibrated']['coverage']:.3f} "
              f"risk {d['recalibrated']['risk']:.3f}")
    print(f"\n[+] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
