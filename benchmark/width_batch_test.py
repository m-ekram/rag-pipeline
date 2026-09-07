import sys
import time
from pathlib import Path

# Project root: F:\pythonprojectsall\rag-faraz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.ocr import PaddleOCRProvider


IMAGE = PROJECT_ROOT / "benchmark_page.png"

CPU_THREADS = 6
BATCH_SIZE = 6

# Widths to test
WIDTHS = [64, 96, 128, 160, 192, 256, 320]


print("=" * 60)
print("PADDLEOCR RECOGNITION WIDTH BENCHMARK")
print("=" * 60)

print("\n[1] Initializing PaddleOCR...")

start = time.perf_counter()

ocr = PaddleOCRProvider(lang="hi")

init_time = time.perf_counter() - start

pipeline = ocr.ocr.paddlex_pipeline._pipeline
rec = pipeline.text_rec_model

print(f"Initialization time: {init_time:.2f} sec")

print("\n[2] Configuration")
print("Recognition model:", rec.model_name)
print("Device:", rec.device)
print("CPU threads:", rec.runner._config.get("cpu_threads"))
print("Batch size:", BATCH_SIZE)

resize_norm = rec.pre_tfs["ReisizeNorm"]

print("Original max_imgW:", resize_norm.max_imgW)
print("Original rec_image_shape:", resize_norm.rec_image_shape)

print("\n[3] Detecting text regions...")

start = time.perf_counter()

detection = list(
    pipeline.text_det_model.predict(
        str(IMAGE),
        batch_size=1
    )
)[0]

detection_time = time.perf_counter() - start

print(f"Detection time: {detection_time:.2f} sec")
print("Detected boxes:", len(detection["dt_polys"]))

print("\n[4] Cropping text regions...")

start = time.perf_counter()

crops = pipeline._crop_by_polys(
    detection["input_img"],
    detection["dt_polys"]
)

crop_time = time.perf_counter() - start

print(f"Crop time: {crop_time:.4f} sec")
print("Crops:", len(crops))


# Width statistics
widths = [crop.shape[1] for crop in crops]
heights = [crop.shape[0] for crop in crops]

print("\n[5] Crop dimensions")
print("Min width: ", min(widths))
print("Max width: ", max(widths))
print("Mean width:", round(sum(widths) / len(widths), 2))

print("Min height: ", min(heights))
print("Max height: ", max(heights))
print("Mean height:", round(sum(heights) / len(heights), 2))


print("\n" + "=" * 60)
print("WIDTH BENCHMARK")
print("=" * 60)

results = []

for width in WIDTHS:

    print(f"\nTesting max_imgW = {width}")

    # Change ONLY the recognition resize width.
    resize_norm.max_imgW = width

    # Keep the model's normal recognition image height.
    # rec_image_shape remains [3, 48, 320].
    start = time.perf_counter()

    predictions = list(
        rec.predict(
            crops,
            batch_size=BATCH_SIZE
        )
    )

    elapsed = time.perf_counter() - start

    results.append(
        {
            "width": width,
            "time": elapsed,
            "predictions": len(predictions),
        }
    )

    print(
        f"Time: {elapsed:.2f} sec | "
        f"Predictions: {len(predictions)}"
    )


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print()
print(f"{'Width':>8} | {'Time (sec)':>12} | {'Predictions':>12}")
print("-" * 42)

for result in results:
    print(
        f"{result['width']:>8} | "
        f"{result['time']:>12.2f} | "
        f"{result['predictions']:>12}"
    )

print("\n" + "=" * 60)

best = min(results, key=lambda x: x["time"])

print("FASTEST")
print("=======")
print("Width:", best["width"])
print("Time:", round(best["time"], 2), "sec")
print("Predictions:", best["predictions"])

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)