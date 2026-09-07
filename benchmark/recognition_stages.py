import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from ingestion.ocr import PaddleOCRProvider


IMAGE_PATH = "benchmark_page.png"


def main():

    print("=" * 60)
    print("PADDLEOCR RECOGNITION INTERNAL STAGE BENCHMARK")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Initialize OCR
    # --------------------------------------------------

    print("\n[1] Initializing PaddleOCR...")

    init_start = time.perf_counter()

    provider = PaddleOCRProvider(lang="hi")

    init_time = time.perf_counter() - init_start

    print(f"Initialization time: {init_time:.2f} sec")

    pipeline = provider.ocr.paddlex_pipeline._pipeline

    detector = pipeline.text_det_model
    recognizer = pipeline.text_rec_model

    # --------------------------------------------------
    # 2. Detection
    # --------------------------------------------------

    print("\n[2] Detecting text regions...")

    start = time.perf_counter()

    detection = list(
        detector.predict(
            IMAGE_PATH,
            batch_size=1,
        )
    )[0]

    detection_time = time.perf_counter() - start

    image = detection["input_img"]
    polygons = detection["dt_polys"]

    print(f"Detection time: {detection_time:.2f} sec")
    print(f"Detected boxes: {len(polygons)}")

    # --------------------------------------------------
    # 3. Crop
    # --------------------------------------------------

    print("\n[3] Cropping text regions...")

    start = time.perf_counter()

    crops = pipeline._crop_by_polys(
        image,
        polygons,
    )

    crop_time = time.perf_counter() - start

    print(f"Crop time: {crop_time:.4f} sec")
    print(f"Crops: {len(crops)}")

    # --------------------------------------------------
    # 4. Read
    # --------------------------------------------------

    print("\n[4] Benchmarking Read stage...")

    start = time.perf_counter()

    read_output = recognizer.pre_tfs["Read"](crops)

    read_time = time.perf_counter() - start

    print(f"Read time: {read_time:.4f} sec")

    # --------------------------------------------------
    # 5. Resize + Normalize
    # --------------------------------------------------

    print("\n[5] Benchmarking Resize + Normalize...")

    start = time.perf_counter()

    resize_output = recognizer.pre_tfs["ReisizeNorm"](
        read_output
    )

    resize_time = time.perf_counter() - start

    print(f"Resize + Normalize time: {resize_time:.4f} sec")

    # --------------------------------------------------
    # 6. ToBatch
    # --------------------------------------------------

    print("\n[6] Benchmarking ToBatch...")

    start = time.perf_counter()

    batch_output = recognizer.pre_tfs["ToBatch"](
        resize_output
    )

    batch_time = time.perf_counter() - start

    print(f"ToBatch time: {batch_time:.4f} sec")

    print("\nModel input batches:", len(batch_output))

    for i, batch in enumerate(batch_output):
        print(
            f"Batch {i}: "
            f"shape={batch.shape}, "
            f"dtype={batch.dtype}"
        )

    # --------------------------------------------------
    # 7. ACTUAL MODEL INFERENCE
    # --------------------------------------------------

    print("\n[7] Benchmarking ACTUAL MODEL INFERENCE...")

    start = time.perf_counter()

    predictions = list(
        recognizer.runner.infer(
            batch_output
        )
    )

    inference_time = time.perf_counter() - start

    print(
        f"Runner inference time: "
        f"{inference_time:.2f} sec"
    )

    print(
        f"Prediction objects: "
        f"{len(predictions)}"
    )

    # --------------------------------------------------
    # 8. Inspect predictions
    # --------------------------------------------------

    print("\n[8] Inspecting model output...")

    if len(predictions) > 0:

        first_prediction = predictions[0]

        print(
            "First prediction type:",
            type(first_prediction),
        )

        if isinstance(first_prediction, dict):

            print(
                "Prediction keys:",
                list(first_prediction.keys()),
            )

        elif hasattr(first_prediction, "shape"):

            print(
                "Prediction shape:",
                first_prediction.shape,
            )

    # --------------------------------------------------
    # 9. CTC post-processing
    # --------------------------------------------------

    print("\n[9] Benchmarking CTC decoding...")

    start = time.perf_counter()

    decoded = recognizer.post_op(
        predictions
    )

    decode_time = time.perf_counter() - start

    print(
        f"CTC decoding time: "
        f"{decode_time:.4f} sec"
    )

    # --------------------------------------------------
    # 10. Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RECOGNITION INTERNAL TIMING")
    print("=" * 60)

    print(f"Read:                 {read_time:.4f} sec")
    print(f"Resize + Normalize:   {resize_time:.4f} sec")
    print(f"ToBatch:              {batch_time:.4f} sec")
    print(f"MODEL INFERENCE:      {inference_time:.2f} sec")
    print(f"CTC decoding:         {decode_time:.4f} sec")

    internal_total = (
        read_time
        + resize_time
        + batch_time
        + inference_time
        + decode_time
    )

    print("-" * 60)
    print(
        f"Internal total:       "
        f"{internal_total:.2f} sec"
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()