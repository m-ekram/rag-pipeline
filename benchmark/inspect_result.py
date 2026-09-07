import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import inspect
from ingestion.ocr import PaddleOCRProvider


print("\n# INSPECT TEXTRECRESULT\n")

print("[1] Initializing PaddleOCR...")
ocr = PaddleOCRProvider(lang="hi")

pipeline = ocr.ocr.paddlex_pipeline._pipeline
rec = pipeline.text_rec_model

print("Recognition model:", rec.model_name)
print("Device:", rec.device)
print("CPU threads:", rec.runner._config.get("cpu_threads"))
print("Batch size:", rec.batch_sampler.batch_size)

print("\n[2] Running OCR on the test PDF...")

# IMPORTANT:
# Replace this with the SAME PDF path used by your previous benchmark.
PDF_PATH = r"F:\pythonprojectsall\rag-faraz\benchmark_page.png"

results = list(ocr.ocr.predict(PDF_PATH))

print("Number of top-level results:", len(results))

if results:
    first = results[0]

    print("\n[3] FIRST RESULT")
    print("Type:", type(first))

    print("\n[4] DIR")
    print([
        x for x in dir(first)
        if not x.startswith("__")
    ])

    print("\n[5] VARS")
    try:
        print(vars(first))
    except Exception as e:
        print("vars() failed:", e)

    print("\n[6] REPRESENTATION")
    print(repr(first))

    print("\n[7] POSSIBLE TEXT ATTRIBUTES")

    for name in [
        "rec_text",
        "text",
        "rec_score",
        "score",
        "input_path",
        "input_img",
    ]:
        try:
            value = getattr(first, name)
            print(f"{name}: {repr(value)}")
        except Exception as e:
            print(f"{name}: <not available> ({e})")

print("\n# DONE\n")