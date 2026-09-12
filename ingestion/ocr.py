import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR output for a single page."""

    text: str
    page: int = 0
    confidence: Optional[float] = 0.95
    engine: str = ""


class OCRProvider(Protocol):
    """Interface implemented by OCR engines."""

    def extract_page(self, image_path: str, page: int) -> OCRResult:
        ...


def remove_table_grid_lines(image: Image.Image) -> Image.Image:
    """
    Strips dense black tabular grid lines using morphological subtraction.
    Prevents OCR detection networks from confusing table borders with digits.
    """
    try:
        import cv2
        import numpy as np

        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Binarize
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Extract horizontal & vertical lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)

        # Subtract table border lines
        table_lines = cv2.add(horiz_lines, vert_lines)
        clean_thresh = cv2.subtract(thresh, table_lines)

        # Invert back to clean black text on white background
        clean_img = cv2.bitwise_not(clean_thresh)
        return Image.fromarray(clean_img)
    except Exception as e:
        logger.debug("Grid line subtraction skipped: %s", e)
        return image


# Recognition model per language. Devanagari covers Hindi and Marathi; the
# Latin model handles English and transliterated text.
_REC_MODELS = {
    "hi": "devanagari_PP-OCRv5_mobile_rec",
    "hin": "devanagari_PP-OCRv5_mobile_rec",
    "mr": "devanagari_PP-OCRv5_mobile_rec",
    "en": "en_PP-OCRv5_mobile_rec",
}


class NoOCREngineAvailable(RuntimeError):
    """Neither PaddleOCR nor Tesseract can run.

    Raised instead of returning empty text: a silent 0-character page looks
    exactly like a blank scan, so a broken install used to surface as
    "No text could be extracted" with no indication of why.
    """


def _run_paddle(engine, img_np) -> tuple[str, list[float]]:
    """Call PaddleOCR across the 2.x and 3.x APIs.

    PaddleOCR 3.x removed `ocr(img, cls=...)` in favour of `predict(img)`, which
    returns per-page dicts with `rec_texts` / `rec_scores`. Calling the 2.x form
    against 3.x raises `unexpected keyword argument 'cls'`, which was being
    swallowed into an empty page.
    """
    predict = getattr(engine, "predict", None)
    if predict is not None:
        texts: list[str] = []
        scores: list[float] = []
        for item in predict(img_np) or []:
            data = getattr(item, "json", None)
            data = data() if callable(data) else data
            if not isinstance(data, dict):
                continue
            payload = data.get("res", data)
            texts.extend(str(t) for t in payload.get("rec_texts", []) if str(t).strip())
            for value in payload.get("rec_scores", []):
                try:
                    scores.append(float(value))
                except (TypeError, ValueError):
                    pass
        if texts:
            return "\n".join(texts).strip(), scores

    legacy = getattr(engine, "ocr", None)
    if legacy is not None:
        result = legacy(img_np, cls=False)
        if result and result[0]:
            lines, scores = [], []
            for line in result[0]:
                if line and line[1]:
                    lines.append(line[1][0])
                    if len(line[1]) > 1:
                        try:
                            scores.append(float(line[1][1]))
                        except (TypeError, ValueError):
                            pass
            return "\n".join(lines).strip(), scores

    return "", []


# Project-local language data (eng, hin, urd, osd), used when the install's own
# tessdata folder lacks Hindi or Urdu.
_PROJECT_TESSDATA = Path(__file__).resolve().parent.parent / ".cache" / "tessdata"
# Where Windows installers put the binary; neither is added to PATH reliably.
_TESSERACT_LOCATIONS = (
    Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
)


def _configure_tesseract(pytesseract) -> str:
    """Point pytesseract at a usable binary; return extra CLI options for language data.

    Without a binary, pytesseract raises on every page and the fallback turned
    that into empty text, so a scanned roll "extracted" as blank pages.
    """
    import shutil

    if not shutil.which(pytesseract.pytesseract.tesseract_cmd):
        found = next((p for p in _TESSERACT_LOCATIONS if p.exists()), None)
        if found is None:
            raise NoOCREngineAvailable(
                "Tesseract is not installed: pytesseract found no tesseract binary. "
                "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
                "(with Hindi/Urdu data), or run where PaddleOCR works."
            )
        pytesseract.pytesseract.tesseract_cmd = str(found)
    if _PROJECT_TESSDATA.is_dir() and not os.environ.get("TESSDATA_PREFIX"):
        return f' --tessdata-dir "{_PROJECT_TESSDATA}"'
    return ""


class RobustPaddleOCREngine:
    """Production Engine for ARM64 CPU:
    - Tier 1: PaddleOCR v4 (Lightweight CNN, ~1.2s/page, high Hindi precision)
    - Tier 2: Native Tesseract (0.3s/page, 100% fail-safe fallback)
    """

    def __init__(self, lang: str = "hi", text_det_unclip_ratio: float = 1.5, **kwargs):
        self.raw_lang = lang
        if "urd" in lang.lower() or "urdu" in lang.lower():
            self.lang = "urd"
        elif "hin" in lang.lower() or "hi" in lang.lower():
            self.lang = "hi"
        else:
            self.lang = "en"
        self.text_det_unclip_ratio = text_det_unclip_ratio
        self.extra_kwargs = kwargs
        self.ocr = None
        self._loaded = False
        self._last_scores: list[float] = []
        self._engine_used = ""
        self._engine_checked = False
        # On Ubuntu 24.04 ARM64, paddlepaddle 3.x has a known glibc std::filesystem C++ ABI segfault.
        # We enable Paddle only if explicitly requested, otherwise default to rock-solid Tesseract.
        # For Urdu, always use Tesseract (with installed tesseract-ocr-urd).
        # paddlepaddle 3.x segfaults on Ubuntu ARM64 (glibc std::filesystem ABI),
        # so it is disabled there by default and enabled everywhere else —
        # defaulting to off globally silently broke macOS and Windows, where
        # PaddleOCR is the only working engine unless Tesseract is installed.
        known_bad = sys.platform.startswith("linux") and platform.machine().lower() in (
            "aarch64", "arm64",
        )
        override = os.environ.get("ENABLE_PADDLEOCR")
        default_on = not known_bad
        self.paddle_available = (
            (override == "1") if override is not None else default_on
        ) and self.lang != "urd"

    def _lazy_load(self):
        if not self._loaded and self.paddle_available and self.lang != "urd":
            try:
                from paddleocr import PaddleOCR

                # Benchmarked configuration. Measured on the Bihar electoral
                # rolls: mobile detection + Devanagari mobile recognition with
                # the three document-analysis pipelines disabled runs ~6s/page
                # at ~0.95 confidence. The generic PP-OCRv4 defaults enable
                # doc-orientation, unwarping and textline-orientation and load
                # server-size detection, which measured 144-195s/page at
                # 0.46-0.77 — roughly 25x slower AND less accurate.
                try:
                    self.ocr = PaddleOCR(
                        lang=self.lang,
                        text_detection_model_name="PP-OCRv5_mobile_det",
                        text_recognition_model_name=_REC_MODELS.get(
                            self.lang, "devanagari_PP-OCRv5_mobile_rec"
                        ),
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        text_det_unclip_ratio=self.text_det_unclip_ratio,
                    )
                except Exception as exc:
                    logger.warning(
                        "Benchmarked OCR config unavailable (%s); falling back to "
                        "PaddleOCR defaults — expect substantially slower pages.", exc
                    )
                    try:
                        self.ocr = PaddleOCR(lang=self.lang)
                    except Exception:
                        self.ocr = PaddleOCR()
                self._loaded = True
            except Exception as e:
                logger.warning("PaddleOCR init failed: %s. Using Tesseract.", e)
                self.paddle_available = False

    def extract_text(self, image: Image.Image) -> str:
        if image is None:
            return ""

        # Per-page state. Without the reset, a page that falls back to Tesseract
        # (or yields nothing) reports the previous Paddle page's confidence.
        self._last_scores = []
        self._engine_used = ""

        # Preprocess with table grid line subtraction for non-Urdu text
        clean_image = remove_table_grid_lines(image) if self.lang != "urd" else image

        # Tier 1: Try Stable PaddleOCR v4
        if self.paddle_available and self.lang != "urd":
            try:
                self._lazy_load()
                if self._loaded and self.ocr is not None:
                    img_np = np.array(clean_image.convert("RGB"))
                    text, scores = _run_paddle(self.ocr, img_np)
                    if text:
                        self._last_scores = scores
                        self._engine_used = "paddleocr"
                        return text
            except Exception as e:
                # A broken Paddle build fails on every page (on Windows, Paddle
                # 3.3's oneDNN path raises "ConvertPirAttribute2RuntimeAttribute
                # not support"); retrying it per page only adds time and log noise.
                self.paddle_available = False
                logger.warning("[!] PaddleOCR error: %s. Using Tier 2 (Tesseract) from now on.", e)

        # Tier 2: Instant Native Tesseract Safety Net (hin+eng / urd / eng)
        try:
            import pytesseract

            tessdata = _configure_tesseract(pytesseract)
            if self.lang == "urd":
                tess_lang = "urd"
            elif "+" in getattr(self, "raw_lang", ""):
                tess_lang = self.raw_lang
            elif "hin" in self.lang or "hi" in self.lang:
                tess_lang = "hin+eng"
            else:
                tess_lang = "eng"
            text = pytesseract.image_to_string(clean_image, lang=tess_lang,
                                               config=f"--psm 6{tessdata}")
            self._engine_used = "tesseract"
            return text.strip()
        except ImportError as exc:
            # No Paddle (or it failed) AND no Tesseract: every page would return
            # "" and the caller would report an empty document with no cause.
            raise NoOCREngineAvailable(
                "No usable OCR engine. PaddleOCR is unavailable or failed, and "
                "pytesseract is not installed. Install one of:\n"
                "  pip install paddleocr paddlepaddle   (then ENABLE_PADDLEOCR=1)\n"
                "  pip install pytesseract  +  the tesseract binary"
            ) from exc
        except NoOCREngineAvailable:
            raise
        except Exception as e:
            logger.warning("[!] Tesseract fallback error: %s", e)
            return ""

    def process_image(self, image: Image.Image) -> OCRResult:
        text = self.extract_text(image)
        return OCRResult(text=text, confidence=0.92)

    def extract_page(self, image_path: str, page: int = 0) -> OCRResult:
        """Compatibility wrapper for PDFExtractor."""
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                text = self.extract_text(img_rgb)
                engine_name = self._engine_used or (
                    "tesseract-native" if (not self._loaded or self.lang == "urd" or not self.paddle_available)
                    else "paddleocr"
                )
                # Real mean recognition score when the engine reports one. The
                # previous constant 0.90 was fabricated: it made a broken engine
                # look confident and corrupted any reported OCR confidence.
                scores = self._last_scores
                confidence = (sum(scores) / len(scores)) if scores else (0.0 if not text else None)
                return OCRResult(
                    text=text,
                    page=page,
                    confidence=confidence,
                    engine=engine_name,
                )
        except NoOCREngineAvailable:
            # A missing engine is a setup error, not an unreadable page: turned
            # into empty text here, it made every scanned PDF look blank.
            raise
        except Exception as e:
            logger.warning("[!] Error reading image %s for OCR: %s", image_path, e)
            return OCRResult(text="", page=page, confidence=0.0, engine="error")


class TesseractOCRProvider(RobustPaddleOCREngine):
    """Pure Tesseract OCR provider (bypasses PaddleOCR completely)."""

    def __init__(self, lang: str = "hin+eng", text_det_unclip_ratio: float = 1.5, **kwargs):
        super().__init__(lang=lang, text_det_unclip_ratio=text_det_unclip_ratio, **kwargs)
        self.paddle_available = False


# Aliases for backward compatibility
PaddleOCRProvider = RobustPaddleOCREngine
ClassicSuryaOCREngine = RobustPaddleOCREngine
ResilientOCREngine = RobustPaddleOCREngine

_ENGINE: Optional[RobustPaddleOCREngine] = None


def ocr_page(image: Image.Image, lang: str = "hi", text_det_unclip_ratio: float = 1.5, **kwargs) -> str:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RobustPaddleOCREngine(lang=lang, text_det_unclip_ratio=text_det_unclip_ratio, **kwargs)
    return _ENGINE.extract_text(image)


def render_pdf_page_to_image(page, target_dpi: int = 150):
    """
    Renders a PyMuPDF (fitz) page at 150 DPI sweet spot.
    Balances crisp text (Devanagari matras, small decimals) with 
    sub-6s CPU compute on ARM64.
    """
    import pymupdf
    # 150 DPI zoom factor relative to standard 72 pt PDF
    zoom = target_dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix