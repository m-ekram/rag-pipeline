import sys
import time
import os
from pathlib import Path

# Project root: F:\pythonprojectsall\rag-faraz
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.ocr import PaddleOCRProvider


IMAGE = PROJECT_ROOT / "benchmark_page.png"

# CPU thread counts to test
THREAD_COUNTS = [1, 2, 4, 6, 8, 10, 12]


print("=" * 60)
print("PADDLEOCR CPU THREAD BENCHMARK")
print("=" * 60)

print("\n[1] Environment")
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))

print("\n[2] Initializing PaddleOCR...")

start = time.perf_counter()

ocr = PaddleOCRProvider(lang="hi")

init_time = time.perf_counter() - start

pipeline = ocr.ocr.paddlex_pipeline._pipeline
rec = pipeline.text_rec_model

print("Initialization time:", round(init_time, 2), "sec")
print("Recognition model:", rec.model_name)
print("Current device:", rec.device)
print("Current Paddle CPU threads:", rec.engine_config.get("cpu_threads"))

print("\n[3] Detecting text regions...")

start = time.perf_counter()

det_result = list(
    pipeline.text_det_model.predict(
        str(IMAGE),
        batch_size=1
    )
)[0]

det_time = time.perf_counter() - start

print("Detection time:", round(det_time, 2), "sec")

crops = pipeline._crop_by_polys(
    det_result["input_img"],
    det_result["dt_polys"]
)

print("Detected boxes:", len(crops))
print("Crops:", len(crops))

print("\n[4] CPU THREAD BENCHMARK")
print("=" * 60)

results = []

for threads in THREAD_COUNTS:

    print(f"\nTesting cpu_threads = {threads}")

    # Change the runner's CPU thread configuration
    rec.engine_config["cpu_threads"] = threads
    rec.runner._config["cpu_threads"] = threads

    # Warm-up
    try:
        list(
            rec.predict(
                crops,
                batch_size=1
            )
        )
    except Exception as e:
        print("Warm-up failed:", repr(e))
        continue

    # Actual measurement
    start = time.perf_counter()

    try:
        predictions = list(
            rec.predict(
                crops,
                batch_size=1
            )
        )

        elapsed = time.perf_counter() - start

        print(
            "Time:",
            round(elapsed, 2),
            "sec | Predictions:",
            len(predictions)
        )

        results.append(
            (threads, elapsed, len(predictions))
        )

    except Exception as e:
        print("Benchmark failed:", repr(e))


print("\n")
print("=" * 60)
print("CPU THREAD RESULTS")
print("=" * 60)

print(f"{'Threads':>10} | {'Time (sec)':>12} | {'Predictions':>12}")
print("-" * 42)

for threads, elapsed, count in results:
    print(
        f"{threads:>10} | "
        f"{elapsed:>12.2f} | "
        f"{count:>12}"
    )

if results:
    best = min(results, key=lambda x: x[1])

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)

    print("Best CPU threads:", best[0])
    print("Best time:", round(best[1], 2), "sec")
    print("Predictions:", best[2])

    baseline = 21.88
    improvement = ((baseline - best[1]) / baseline) * 100

    print(
        "Improvement vs current 21.88 sec baseline:",
        round(improvement, 2),
        "%"
    )

print("\nDONE")
print("=" * 60)