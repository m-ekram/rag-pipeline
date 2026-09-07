import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.ocr import PaddleOCRProvider


def describe(name, value):
    """Print useful information about an object."""
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Type:", type(value))

    if isinstance(value, dict):
        print("Dict keys:", list(value.keys()))

    elif isinstance(value, (list, tuple)):
        print("Length:", len(value))

        if len(value) > 0:
            print("First item type:", type(value[0]))

            if hasattr(value[0], "shape"):
                print("First item shape:", value[0].shape)

    elif hasattr(value, "shape"):
        print("Shape:", value.shape)

    print("\nValue preview:")

    try:
        print(str(value)[:500])
    except Exception as e:
        print("Could not print value:", e)


def main():

    print("=" * 60)
    print("PADDLEOCR RECOGNITION DATA FLOW INSPECTION")
    print("=" * 60)

    # --------------------------------------------------
    # Initialize OCR
    # --------------------------------------------------

    print("\n[1] Initializing PaddleOCR...")

    provider = PaddleOCRProvider(lang="hi")

    pipeline = provider.ocr.paddlex_pipeline._pipeline

    detector = pipeline.text_det_model
    recognizer = pipeline.text_rec_model

    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    print("\n[2] Detecting text regions...")

    detection = list(
        detector.predict(
            "benchmark_page.png",
            batch_size=1,
        )
    )[0]

    image = detection["input_img"]
    polygons = detection["dt_polys"]

    print("Detected boxes:", len(polygons))

    # --------------------------------------------------
    # Cropping
    # --------------------------------------------------

    print("\n[3] Cropping text regions...")

    crops = pipeline._crop_by_polys(
        image,
        polygons,
    )

    print("Crops:", len(crops))

    describe(
        "ORIGINAL CROPS",
        crops,
    )

    # Use only one crop first to understand the flow
    sample = [crops[0]]

    # --------------------------------------------------
    # Stage 1: Read
    # --------------------------------------------------

    print("\n[4] Read stage...")

    read_output = recognizer.pre_tfs["Read"](sample)

    describe(
        "AFTER READ",
        read_output,
    )

    # --------------------------------------------------
    # Stage 2: Resize + Normalize
    # --------------------------------------------------

    print("\n[5] Resize + Normalize stage...")

    resize_output = recognizer.pre_tfs["ReisizeNorm"](read_output)

    describe(
        "AFTER RESIZE + NORMALIZE",
        resize_output,
    )

    # --------------------------------------------------
    # Stage 3: ToBatch
    # --------------------------------------------------

    print("\n[6] ToBatch stage...")

    batch_output = recognizer.pre_tfs["ToBatch"](resize_output)

    describe(
        "AFTER TOBATCH",
        batch_output,
    )

    # --------------------------------------------------
    # Stage 4: Runner inference
    # --------------------------------------------------

    print("\n[7] Inspecting inference input...")

    if isinstance(batch_output, list):

        for i, item in enumerate(batch_output):

            print(f"\nBatch item {i}:")
            print("Type:", type(item))

            if hasattr(item, "shape"):
                print("Shape:", item.shape)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()