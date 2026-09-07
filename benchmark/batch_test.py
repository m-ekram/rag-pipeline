import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)



import time

from ingestion.ocr import PaddleOCRProvider


IMAGE_PATH = "benchmark_page.png"


def main():
    print("=" * 60)
    print("PADDLEOCR RECOGNITION BATCH BENCHMARK")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Initialize OCR
    # --------------------------------------------------

    print("\n[1] Initializing PaddleOCR...")

    start = time.perf_counter()

    provider = PaddleOCRProvider(lang="hi")

    init_time = time.perf_counter() - start

    print(f"Initialization time: {init_time:.2f} sec")

    # --------------------------------------------------
    # 2. Access internal pipeline
    # --------------------------------------------------

    pipeline = provider.ocr.paddlex_pipeline._pipeline

    det_model = pipeline.text_det_model
    rec_model = pipeline.text_rec_model

    print("\n[2] Internal models found")
    print("Detection:", type(det_model))
    print("Recognition:", type(rec_model))

    # --------------------------------------------------
    # 3. Detection
    # --------------------------------------------------

    print("\n[3] Detecting text boxes...")

    start = time.perf_counter()

    detection = list(
        det_model.predict(
            IMAGE_PATH,
            batch_size=1,
        )
    )[0]

    detection_time = time.perf_counter() - start

    image = detection["input_img"]
    polys = detection["dt_polys"]

    print(f"Detection time: {detection_time:.2f} sec")
    print(f"Detected boxes: {len(polys)}")

    # --------------------------------------------------
    # 4. Crop detected regions
    # --------------------------------------------------

    print("\n[4] Cropping detected text regions...")

    start = time.perf_counter()

    crops = pipeline._crop_by_polys(
        image,
        polys,
    )

    crop_time = time.perf_counter() - start

    print(f"Crop time: {crop_time:.3f} sec")
    print(f"Crops: {len(crops)}")

    # --------------------------------------------------
    # 5. Test recognition batch sizes
    # --------------------------------------------------

    batch_sizes = [1, 2, 4, 8, 16, 32, 74]

    print("\n[5] Recognition batch-size benchmark")
    print("-" * 60)

    results = []

    for batch_size in batch_sizes:

        print(f"\nTesting batch_size = {batch_size}")

        start = time.perf_counter()

        recognition_results = list(
            rec_model.predict(
                crops,
                batch_size=batch_size,
            )
        )

        elapsed = time.perf_counter() - start

        results.append(
            (
                batch_size,
                elapsed,
                len(recognition_results),
            )
        )

        print(
            f"Time: {elapsed:.2f} sec | "
            f"Results: {len(recognition_results)}"
        )

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("BATCH SIZE RESULTS")
    print("=" * 60)

    print(
        f"{'Batch':>8} | "
        f"{'Time (sec)':>12} | "
        f"{'Results':>10}"
    )

    print("-" * 60)

    for batch_size, elapsed, count in results:
        print(
            f"{batch_size:>8} | "
            f"{elapsed:>12.2f} | "
            f"{count:>10}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()