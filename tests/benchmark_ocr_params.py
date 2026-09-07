"""Benchmark OCR Parameters on Electoral Rolls.

Compares:
- render_scale: 1.25, 1.50, 1.75, 2.00
- unclip_ratio: 1.5 vs 1.8

Measures:
- Render latency (ms)
- OCR inference latency (s)
- Image dimensions (width x height)
- Total detected bounding boxes
- EPIC detection count (out of 30 voter cards)
- Matra/Keyword fidelity for Hindi fields
- Mean recognition confidence score
"""

import sys
import os
import time
import re
import tempfile
from pathlib import Path
import pymupdf

sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ingestion.ocr import PaddleOCRProvider

EPIC_RE = re.compile(r"[A-Z]{3}[0-9]{7}|BR/\d{2}/\d{3}/\d{6}")

KEYWORDS = {
    "निर्वाचक का नाम": re.compile(r"निर्वाचक\s*का\s*नाम"),
    "पिता का नाम": re.compile(r"पिता\s*का\s*नाम"),
    "मकान संख्या": re.compile(r"मकान\s*संख्या"),
    "लिंग": re.compile(r"लिंग"),
    "आयु": re.compile(r"आयु"),
}


def run_benchmark():
    pdf_path = "data/183/2025-EROLLGEN-S04-183-SIR-FinalRoll-Revision1-HIN-1-WI.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}", flush=True)
        return

    page_num = 3  # Standard voter roll page (30 cards)
    print("=" * 80, flush=True)
    print(f"[*] Benchmarking OCR on: {os.path.basename(pdf_path)} (Page {page_num})", flush=True)
    print("=" * 80, flush=True)

    # We test:
    # 1. Baseline: scale=1.25, unclip=1.5
    # 2. Unclip boost: scale=1.25, unclip=1.8
    # 3. Resolution boost: scale=1.50, unclip=1.5
    # 4. Combined sweet-spot: scale=1.50, unclip=1.8
    # 5. Ultra resolution: scale=2.00, unclip=1.8
    test_cases = [
        {"scale": 1.25, "unclip": 1.5, "label": "Baseline (Current)"},
        {"scale": 1.25, "unclip": 1.8, "label": "Unclip 1.8 Only"},
        {"scale": 1.50, "unclip": 1.5, "label": "Scale 1.5 Only"},
        {"scale": 1.50, "unclip": 1.8, "label": "Scale 1.5 + Unclip 1.8"},
        {"scale": 2.00, "unclip": 1.8, "label": "Scale 2.0 + Unclip 1.8"},
    ]

    doc = pymupdf.open(pdf_path)
    page = doc[page_num - 1]

    # Pre-render images for each scale
    images = {}
    scales_needed = sorted(set(tc["scale"] for tc in test_cases))
    for s in scales_needed:
        t0 = time.perf_counter()
        pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), alpha=False)
        render_ms = (time.perf_counter() - t0) * 1000
        with tempfile.NamedTemporaryFile(suffix=f"_scale_{s}.png", delete=False) as tmp:
            tmp_name = tmp.name
        pix.save(tmp_name)
        images[s] = {
            "path": tmp_name,
            "render_ms": round(render_ms, 1),
            "width": pix.width,
            "height": pix.height,
            "size_kb": round(os.path.getsize(tmp_name) / 1024, 1),
        }
        print(f"[+] Pre-rendered scale {s}x: {pix.width}x{pix.height} ({images[s]['size_kb']} KB) in {render_ms:.1f}ms", flush=True)
    doc.close()

    # Reuse providers per unclip ratio
    providers = {}
    results = []

    try:
        for idx, tc in enumerate(test_cases, 1):
            s = tc["scale"]
            u = tc["unclip"]
            lbl = tc["label"]
            print(f"\n[{idx}/{len(test_cases)}] Testing {lbl} (scale={s}, unclip={u})...", flush=True)

            if u not in providers:
                print(f"    [*] Loading PaddleOCR with unclip_ratio={u}...", flush=True)
                providers[u] = PaddleOCRProvider(lang="hi", text_det_unclip_ratio=u)

            ocr_provider = providers[u]
            img_info = images[s]

            t_start = time.perf_counter()
            ocr_res = ocr_provider.extract_page(img_info["path"], page=page_num)
            ocr_s = time.perf_counter() - t_start

            full_text = ocr_res.text
            epics = set(EPIC_RE.findall(full_text))
            boxes = [l for l in full_text.splitlines() if l.strip()]

            kw_matches = {name: len(pattern.findall(full_text)) for name, pattern in KEYWORDS.items()}

            res = {
                "label": lbl,
                "scale": s,
                "unclip": u,
                "dimensions": f"{img_info['width']}x{img_info['height']}",
                "render_ms": img_info["render_ms"],
                "ocr_s": round(ocr_s, 2),
                "total_s": round((img_info["render_ms"] / 1000) + ocr_s, 2),
                "boxes": len(boxes),
                "epics": len(epics),
                "conf": round(ocr_res.confidence or 0.0, 4),
                "voter_name_kw": kw_matches.get("निर्वाचक का नाम", 0),
                "house_kw": kw_matches.get("मकान संख्या", 0),
                "gender_kw": kw_matches.get("लिंग", 0),
                "age_kw": kw_matches.get("आयु", 0),
            }
            results.append(res)
            print(
                f"    -> Done in {res['ocr_s']}s | EPICs: {res['epics']}/30 | Boxes: {res['boxes']} | "
                f"निर्वाचक का नाम: {res['voter_name_kw']}/30 | मकान संख्या: {res['house_kw']}/30 | Conf: {res['conf']:.1%}",
                flush=True,
            )

    finally:
        for info in images.values():
            if os.path.exists(info["path"]):
                os.remove(info["path"])

    print("\n" + "=" * 95, flush=True)
    print("FINAL BENCHMARK COMPARISON TABLE", flush=True)
    print("=" * 95, flush=True)
    header = (
        f"| {'Configuration':<24} | {'Resolution':<11} | {'Latency':<8} | "
        f"{'EPICs (30)':<10} | {'Boxes':<5} | {'निर्वाचक का नाम':<14} | {'मकान संख्या':<11} | {'Conf':<6} |"
    )
    sep = "|" + "|".join(["-" * (len(col) + 2) for col in header.split("|")[1:-1]]) + "|"
    print(header, flush=True)
    print(sep, flush=True)
    for r in results:
        print(
            f"| {r['label']:<24} | {r['dimensions']:<11} | {r['total_s']:<6.2f}s | "
            f"{r['epics']:<10} | {r['boxes']:<5} | {r['voter_name_kw']:<14} | "
            f"{r['house_kw']:<11} | {r['conf']:<6.1%} |",
            flush=True,
        )
    print("=" * 95, flush=True)


if __name__ == "__main__":
    run_benchmark()
