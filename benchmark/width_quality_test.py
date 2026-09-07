import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.ocr import PaddleOCRProvider


IMAGE = PROJECT_ROOT / "benchmark_page.png"

WIDTHS = [3200, 256]
BATCH_SIZE = 6


print("=" * 70)
print("PADDLEOCR WIDTH QUALITY COMPARISON")
print("=" * 70)

print("\n[1] Initializing PaddleOCR...")

start = time.perf_counter()

ocr = PaddleOCRProvider(lang="hi")

print(f"Initialization time: {time.perf_counter() - start:.2f} sec")

pipeline = ocr.ocr.paddlex_pipeline._pipeline
rec = pipeline.text_rec_model
resize_norm = rec.pre_tfs["ReisizeNorm"]

print("\nRecognition model:", rec.model_name)
print("Device:", rec.device)
print("Internal CPU threads:", rec.runner._config.get("cpu_threads"))
print("Batch size:", BATCH_SIZE)


# ---------------------------------------------------------
# Detection
# ---------------------------------------------------------

print("\n[2] Detecting text regions...")

start = time.perf_counter()

detection = list(
    pipeline.text_det_model.predict(
        str(IMAGE),
        batch_size=1
    )
)[0]

print(f"Detection time: {time.perf_counter() - start:.2f} sec")

crops = pipeline._crop_by_polys(
    detection["input_img"],
    detection["dt_polys"]
)

print("Detected boxes:", len(crops))


# ---------------------------------------------------------
# Recognition
# ---------------------------------------------------------

all_results = {}


for width in WIDTHS:

    print("\n" + "=" * 70)
    print(f"TESTING max_imgW = {width}")
    print("=" * 70)

    resize_norm.max_imgW = width

    start = time.perf_counter()

    predictions = list(
        rec.predict(
            crops,
            batch_size=BATCH_SIZE
        )
    )

    elapsed = time.perf_counter() - start

    print(f"Recognition time: {elapsed:.2f} sec")
    print("Prediction objects:", len(predictions))

    # -----------------------------------------------------
    # Decode predictions using PaddleOCR's actual
    # postprocessor.
    # -----------------------------------------------------

    decoded = []

    for prediction in predictions:

        # PaddleX may return numpy arrays directly.
        if hasattr(prediction, "shape"):

            try:
                result = rec.post_op(prediction)
            except Exception:
                result = None

            decoded.append(result)

        else:
            decoded.append(prediction)

    all_results[width] = {
        "predictions": predictions,
        "decoded": decoded,
        "time": elapsed,
    }


# ---------------------------------------------------------
# Inspect raw prediction objects
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION STRUCTURE")
print("=" * 70)

for width in WIDTHS:

    predictions = all_results[width]["predictions"]

    print(f"\nWidth {width}")

    if predictions:

        first = predictions[0]

        print("First prediction type:", type(first))

        if hasattr(first, "shape"):
            print("Shape:", first.shape)

        if isinstance(first, dict):
            print("Keys:", list(first.keys()))


# ---------------------------------------------------------
# Save raw comparison
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TIMING SUMMARY")
print("=" * 70)

for width in WIDTHS:

    data = all_results[width]

    print(
        f"Width {width:4d} | "
        f"{data['time']:8.2f} sec | "
        f"{len(data['predictions']):3d} predictions"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)