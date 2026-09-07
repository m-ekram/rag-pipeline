import sys
import time
import json
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# This file is:
# F:\pythonprojectsall\rag-faraz\benchmark\width_quality_compare.py
#
# Therefore:
# parent      = benchmark
# parent.parent = rag-faraz

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ingestion.ocr import PaddleOCRProvider


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE = PROJECT_ROOT / "benchmark_page.png"

# Multi-width benchmark
WIDTHS = [
    256,
    512,
    768,
    1024,
    1536,
    2048,
    3200,
]

# Batch size fixed at 1.
# Previous benchmark showed batch size 1 is optimal.
BATCH_SIZE = 1


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PADDLEOCR MULTI-WIDTH TEXT QUALITY COMPARISON")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Test image:")
print(IMAGE)

print()
print("Widths:")
print(WIDTHS)

print()
print("Batch size:")
print(BATCH_SIZE)


# ============================================================
# VALIDATE IMAGE
# ============================================================

if not IMAGE.exists():
    raise FileNotFoundError(
        f"Test image not found:\n{IMAGE}"
    )


# ============================================================
# INITIALIZE OCR
# ============================================================

print()
print("[1] Initializing PaddleOCR...")
print()

start = time.perf_counter()

ocr = PaddleOCRProvider(lang="hi")

init_time = time.perf_counter() - start

print(f"Initialization time: {init_time:.2f} sec")


# ============================================================
# ACCESS INTERNAL PADDLEOCR PIPELINE
# ============================================================

pipeline = ocr.ocr.paddlex_pipeline._pipeline

rec = pipeline.text_rec_model

resize_norm = rec.pre_tfs["ReisizeNorm"]


# ============================================================
# MODEL INFORMATION
# ============================================================

print()
print("Recognition model:")
print(rec.model_name)

print()
print("Device:")
print(rec.device)

print()
print("Internal CPU threads:")
print(rec.runner._config.get("cpu_threads"))

print()
print("Batch size:")
print(BATCH_SIZE)

print()
print("Initial max_imgW:")
print(resize_norm.max_imgW)


# ============================================================
# DETECTION
# ============================================================

print()
print("=" * 70)
print("[2] Detecting text regions...")
print("=" * 70)

start = time.perf_counter()

detection = list(
    pipeline.text_det_model.predict(
        str(IMAGE),
        batch_size=1,
    )
)[0]

detection_time = time.perf_counter() - start

print(f"Detection time: {detection_time:.2f} sec")


# ============================================================
# CROPPING
# ============================================================

print()
print("=" * 70)
print("[3] Cropping detected text regions...")
print("=" * 70)

crops = pipeline._crop_by_polys(
    detection["input_img"],
    detection["dt_polys"],
)

print("Detected boxes:", len(crops))


# ============================================================
# HELPER
# ============================================================

def extract_prediction_data(prediction):
    """
    Extract recognized text and confidence from PaddleX/PaddleOCR
    prediction objects.

    Returns:
        tuple[str, float | None]
    """

    text = ""
    confidence = None

    # --------------------------------------------------------
    # Dictionary prediction
    # --------------------------------------------------------

    if isinstance(prediction, dict):

        possible_text_keys = [
            "rec_text",
            "rec_texts",
            "text",
            "texts",
        ]

        for key in possible_text_keys:

            if key not in prediction:
                continue

            value = prediction[key]

            if isinstance(value, list):

                if value:
                    text = str(value[0])

            else:
                text = str(value)

            if text.strip():
                break

        possible_score_keys = [
            "rec_score",
            "rec_scores",
            "score",
            "scores",
        ]

        for key in possible_score_keys:

            if key not in prediction:
                continue

            value = prediction[key]

            try:

                if isinstance(value, list):

                    if value:
                        confidence = float(value[0])

                else:
                    confidence = float(value)

            except (TypeError, ValueError):
                pass

            if confidence is not None:
                break

        return text.strip(), confidence

    # --------------------------------------------------------
    # PaddleX result object
    # --------------------------------------------------------

    data = getattr(prediction, "json", None)

    if callable(data):

        try:
            data = data()
        except Exception:
            data = None

    if isinstance(data, dict):

        payload = data.get("res", data)

        rec_texts = payload.get(
            "rec_texts",
            [],
        )

        rec_scores = payload.get(
            "rec_scores",
            [],
        )

        if rec_texts:

            text = str(
                rec_texts[0]
            ).strip()

        if rec_scores:

            try:
                confidence = float(
                    rec_scores[0]
                )
            except (
                TypeError,
                ValueError,
            ):
                confidence = None

        return text, confidence

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return "", None


# ============================================================
# RUN MULTI-WIDTH BENCHMARK
# ============================================================

print()
print("=" * 70)
print("[4] Running multi-width comparison...")
print("=" * 70)


all_results = {}


for width in WIDTHS:

    print()
    print("=" * 70)
    print(f"TESTING max_imgW = {width}")
    print("=" * 70)

    # --------------------------------------------------------
    # Change recognition resize width
    # --------------------------------------------------------

    resize_norm.max_imgW = width

    print()
    print("Internal max_imgW:")
    print(resize_norm.max_imgW)

    print()
    print("Batch size:")
    print(BATCH_SIZE)

    # --------------------------------------------------------
    # Recognition
    # --------------------------------------------------------

    start = time.perf_counter()

    predictions = list(
        rec.predict(
            crops,
            batch_size=BATCH_SIZE,
        )
    )

    elapsed = time.perf_counter() - start

    # --------------------------------------------------------
    # Extract text and confidence
    # --------------------------------------------------------

    texts = []
    confidences = []

    for prediction in predictions:

        text, confidence = extract_prediction_data(
            prediction
        )

        texts.append(text)

        if confidence is not None:
            confidences.append(confidence)

    recognized_texts = [
        text
        for text in texts
        if text.strip()
    ]

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    all_results[width] = {
        "width": width,
        "time_sec": elapsed,
        "prediction_count": len(predictions),
        "recognized_text_count": len(
            recognized_texts
        ),
        "average_confidence": average_confidence,
        "texts": texts,
        "confidences": confidences,
    }

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print(
        f"Recognition time: "
        f"{elapsed:.2f} sec"
    )

    print(
        "Prediction objects:",
        len(predictions),
    )

    print(
        "Recognized text entries:",
        len(recognized_texts),
    )

    if average_confidence is not None:

        print(
            f"Average confidence: "
            f"{average_confidence:.4f}"
        )

    else:

        print(
            "Average confidence: N/A"
        )


# ============================================================
# WIDTH SUMMARY
# ============================================================

print()
print("=" * 70)
print("WIDTH COMPARISON SUMMARY")
print("=" * 70)

print()

print(
    f"{'WIDTH':>8} | "
    f"{'TIME (sec)':>12} | "
    f"{'PREDICTIONS':>12} | "
    f"{'TEXTS':>8} | "
    f"{'AVG CONF':>10}"
)

print("-" * 70)

for width in WIDTHS:

    data = all_results[width]

    confidence = data[
        "average_confidence"
    ]

    confidence_text = (
        f"{confidence:.4f}"
        if confidence is not None
        else "N/A"
    )

    print(
        f"{width:>8} | "
        f"{data['time_sec']:>12.2f} | "
        f"{data['prediction_count']:>12} | "
        f"{data['recognized_text_count']:>8} | "
        f"{confidence_text:>10}"
    )


# ============================================================
# FIND FASTEST WIDTH
# ============================================================

fastest_width = min(
    WIDTHS,
    key=lambda w: all_results[w]["time_sec"],
)

fastest_time = all_results[
    fastest_width
]["time_sec"]


# ============================================================
# FIND HIGHEST CONFIDENCE WIDTH
# ============================================================

valid_confidence_widths = [
    width
    for width in WIDTHS
    if all_results[width][
        "average_confidence"
    ] is not None
]

if valid_confidence_widths:

    highest_confidence_width = max(
        valid_confidence_widths,
        key=lambda w: all_results[w][
            "average_confidence"
        ],
    )

    highest_confidence = all_results[
        highest_confidence_width
    ]["average_confidence"]

else:

    highest_confidence_width = None
    highest_confidence = None


# ============================================================
# TEXT COMPARISON
# ============================================================

print()
print("=" * 70)
print("TEXT COMPARISON")
print("=" * 70)


num_regions = len(crops)


for region_index in range(num_regions):

    print()
    print("-" * 70)
    print(
        f"TEXT REGION #{region_index + 1}"
    )
    print("-" * 70)

    for width in WIDTHS:

        data = all_results[width]

        texts = data["texts"]

        # IMPORTANT:
        # confidences is a separate compact list containing
        # only valid confidence values. Therefore using the
        # same index can become misaligned if a confidence is
        # missing.
        #
        # We handle that safely below.

        text = (
            texts[region_index]
            if region_index < len(texts)
            else ""
        )

        confidence = None

        if region_index < len(
            data["confidences"]
        ):

            confidence = data[
                "confidences"
            ][region_index]

        print()
        print(
            f"Width {width}:"
        )

        if confidence is not None:

            print(
                f"Confidence: "
                f"{confidence:.4f}"
            )

        else:

            print(
                "Confidence: N/A"
            )

        print(
            f"Text: {text}"
        )


# ============================================================
# SPEED COMPARISON
# ============================================================

baseline_width = 3200

baseline_time = all_results[
    baseline_width
]["time_sec"]


print()
print("=" * 70)
print("SPEED COMPARISON AGAINST WIDTH 3200")
print("=" * 70)

print()

print(
    f"{'WIDTH':>8} | "
    f"{'TIME':>10} | "
    f"{'SPEEDUP':>10} | "
    f"{'TIME SAVED':>12}"
)

print("-" * 55)


for width in WIDTHS:

    current_time = all_results[
        width
    ]["time_sec"]

    speedup = (
        baseline_time / current_time
        if current_time > 0
        else 0
    )

    time_saved = (
        baseline_time - current_time
    )

    print(
        f"{width:>8} | "
        f"{current_time:>10.2f} | "
        f"{speedup:>9.2f}x | "
        f"{time_saved:>11.2f}s"
    )


# ============================================================
# CONFIDENCE COMPARISON
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE COMPARISON")
print("=" * 70)

print()

print(
    f"{'WIDTH':>8} | "
    f"{'AVG CONF':>10} | "
    f"{'DELTA VS 3200':>15}"
)

print("-" * 45)

baseline_confidence = all_results[
    baseline_width
]["average_confidence"]


for width in WIDTHS:

    confidence = all_results[
        width
    ]["average_confidence"]

    if (
        confidence is not None
        and baseline_confidence is not None
    ):

        delta = (
            confidence
            - baseline_confidence
        )

        print(
            f"{width:>8} | "
            f"{confidence:>10.4f} | "
            f"{delta:>+15.4f}"
        )

    else:

        print(
            f"{width:>8} | "
            f"{'N/A':>10} | "
            f"{'N/A':>15}"
        )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("BENCHMARK INTERPRETATION")
print("=" * 70)

print()

print(
    f"Fastest width: "
    f"{fastest_width}"
)

print(
    f"Fastest recognition time: "
    f"{fastest_time:.2f} sec"
)

if highest_confidence_width is not None:

    print()
    print(
        f"Highest average confidence width: "
        f"{highest_confidence_width}"
    )

    print(
        f"Highest average confidence: "
        f"{highest_confidence:.4f}"
    )

print()

print(
    "Batch size was fixed at 1 "
    "for every width."
)

print()

print(
    "Widths tested: "
    "256 → 512 → 768 → 1024 → "
    "1536 → 2048 → 3200"
)

print()

print(
    "The purpose of this benchmark is to "
    "identify the smallest max_imgW that "
    "provides acceptable Hindi OCR quality "
    "while reducing recognition time."
)


# ============================================================
# SAVE RESULTS
# ============================================================

print()
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)


results_path = (
    PROJECT_ROOT
    / "benchmark"
    / "width_quality_results.json"
)


results_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


json_results = {
    "test_image": str(IMAGE),
    "model": rec.model_name,
    "device": str(rec.device),
    "cpu_threads": rec.runner._config.get(
        "cpu_threads"
    ),
    "batch_size": BATCH_SIZE,
    "widths": WIDTHS,
    "detection_time_sec": detection_time,
    "detected_boxes": len(crops),
    "fastest_width": fastest_width,
    "fastest_time_sec": fastest_time,
    "highest_confidence_width": (
        highest_confidence_width
    ),
    "highest_confidence": (
        highest_confidence
    ),
    "results": {
        str(width): all_results[width]
        for width in WIDTHS
    },
}


with open(
    results_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_results,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("Saved:")
print(results_path)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()

print(
    "Multi-width benchmark completed "
    "with batch size 1."
)

print(
    "Widths tested:",
    " → ".join(
        str(w)
        for w in WIDTHS
    ),
)

print()