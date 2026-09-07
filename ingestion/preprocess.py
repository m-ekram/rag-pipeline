"""Preprocessing and adaptive reconciliation for dual PDF extraction.

The ingestion system may obtain text from:
1. Native PDF text extraction
2. OCR

This module evaluates native extraction using multiple signals and decides
whether OCR is necessary before text enters the normal RAG pipeline.
"""

from dataclasses import dataclass
import re
import unicodedata

# Devanagari (and most Indic scripts) write vowels as combining marks — Unicode
# categories Mn/Mc. Python's str.isalpha() is False for those, so a clean Hindi
# page scored alpha_ratio ~0.48 and suspicious_ratio ~0.32 against ~0.83/~0.01
# for the same text in English. That tripped `suspicious_ratio > 0.15` and
# forced OCR on text that was already perfect. Treating marks as part of the
# letter they attach to fixes the scores for every Indic script at once.
_LETTER_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})


def _is_letter(char: str) -> bool:
    """True for letters and the combining marks that belong to them."""
    return char.isalpha() or unicodedata.category(char) in _LETTER_MARK_CATEGORIES


@dataclass(frozen=True)
class ExtractionResult:
    """Text extracted from one PDF page."""

    text: str
    method: str
    page: int
    confidence: float | None = None


@dataclass(frozen=True)
class PreprocessedText:
    """Final text selected for downstream ingestion."""

    text: str
    method: str
    page: int
    native_quality: float
    ocr_quality: float


@dataclass(frozen=True)
class NativeQualityReport:
    """Multi-signal assessment of native PDF extraction."""

    score: float
    text_length: int
    word_count: int
    alpha_ratio: float
    printable_ratio: float
    suspicious_ratio: float
    has_repeated_glyphs: bool
    has_reasonable_structure: bool
    needs_ocr: bool


def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph structure."""

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def text_quality(text: str) -> float:
    """Legacy lightweight text-quality score."""

    text = normalize_text(text)

    if not text:
        return 0.0

    characters = len(text)

    words = re.findall(
        r"\b\w+\b",
        text,
        flags=re.UNICODE,
    )

    if not words:
        return 0.0

    printable_ratio = sum(
        1
        for char in text
        if char.isprintable() or char in "\n\t"
    ) / max(len(text), 1)

    alpha_ratio = sum(
        1
        for char in text
        if _is_letter(char)
    ) / max(characters, 1)

    word_count_score = min(len(words) / 100.0, 1.0)

    score = (
        0.35 * printable_ratio
        + 0.35 * alpha_ratio
        + 0.30 * word_count_score
    )

    return round(
        min(max(score, 0.0), 1.0),
        4,
    )


def _suspicious_ratio(text: str) -> float:
    """Estimate how much extracted text looks corrupted or unusable."""

    if not text:
        return 1.0

    suspicious = 0

    for char in text:
        if char.isalnum() or char.isspace() or _is_letter(char):
            continue

        if char in ".,;:!?()[]{}-_/|":
            continue

        suspicious += 1

    return suspicious / max(len(text), 1)


def _has_repeated_glyphs(text: str) -> bool:
    """Detect pathological repeated-character extraction."""

    if len(text) < 20:
        return False

    compact = re.sub(r"\s+", "", text)

    if not compact:
        return False

    for char in set(compact):
        if compact.count(char) / len(compact) > 0.35:
            return True

    return False


def _has_reasonable_structure(text: str) -> bool:
    """Check whether extracted text has basic document structure."""

    if not text.strip():
        return False

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    if len(text) > 500 and len(lines) == 1:
        return False

    if len(lines) >= 2:
        return True

    return len(text) >= 80


def analyze_native_text(
    text: str,
    *,
    native_threshold: float = 0.55,
) -> NativeQualityReport:
    """Evaluate native extraction using multiple quality signals."""

    text = normalize_text(text)

    if not text:
        return NativeQualityReport(
            score=0.0,
            text_length=0,
            word_count=0,
            alpha_ratio=0.0,
            printable_ratio=0.0,
            suspicious_ratio=1.0,
            has_repeated_glyphs=False,
            has_reasonable_structure=False,
            needs_ocr=True,
        )

    characters = len(text)

    words = re.findall(
        r"\b\w+\b",
        text,
        flags=re.UNICODE,
    )

    word_count = len(words)

    printable_ratio = sum(
        1
        for char in text
        if char.isprintable() or char in "\n\t"
    ) / max(characters, 1)

    alpha_ratio = sum(
        1
        for char in text
        if _is_letter(char)
    ) / max(characters, 1)

    suspicious_ratio = _suspicious_ratio(text)

    repeated_glyphs = _has_repeated_glyphs(text)

    reasonable_structure = _has_reasonable_structure(text)

    length_score = min(characters / 500.0, 1.0)

    word_score = min(word_count / 100.0, 1.0)

    suspicious_score = 1.0 - suspicious_ratio

    structure_score = (
        1.0
        if reasonable_structure
        else 0.0
    )

    repetition_score = (
        0.0
        if repeated_glyphs
        else 1.0
    )

    score = (
        0.25 * length_score
        + 0.20 * word_score
        + 0.20 * alpha_ratio
        + 0.15 * printable_ratio
        + 0.10 * suspicious_score
        + 0.05 * structure_score
        + 0.05 * repetition_score
    )

    score = round(
        min(max(score, 0.0), 1.0),
        4,
    )

    needs_ocr = (
        score < native_threshold
        or characters < 80
        or word_count < 10
        or suspicious_ratio > 0.15
        or repeated_glyphs
        or not reasonable_structure
    )

    return NativeQualityReport(
        score=score,
        text_length=characters,
        word_count=word_count,
        alpha_ratio=round(alpha_ratio, 4),
        printable_ratio=round(printable_ratio, 4),
        suspicious_ratio=round(suspicious_ratio, 4),
        has_repeated_glyphs=repeated_glyphs,
        has_reasonable_structure=reasonable_structure,
        needs_ocr=needs_ocr,
    )


def should_use_ocr(
    text: str,
    *,
    native_threshold: float = 0.55,
) -> NativeQualityReport:
    """Return the complete OCR decision report."""

    return analyze_native_text(
        text,
        native_threshold=native_threshold,
    )


def reconcile_extractions(
    native: ExtractionResult | None,
    ocr: ExtractionResult | None,
    *,
    native_threshold: float = 0.55,
    ocr_threshold: float = 0.30,
) -> PreprocessedText:
    """Choose the best extraction after native/OCR processing."""

    native_text = normalize_text(
        native.text
    ) if native else ""

    ocr_text = normalize_text(
        ocr.text
    ) if ocr else ""

    native_score = text_quality(native_text)
    ocr_score = text_quality(ocr_text)

    page = (
        native.page
        if native is not None
        else ocr.page
        if ocr is not None
        else 0
    )

    if not native_text and not ocr_text:
        return PreprocessedText(
            text="",
            method="none",
            page=page,
            native_quality=native_score,
            ocr_quality=ocr_score,
        )

    if native_text and native_score >= native_threshold:
        return PreprocessedText(
            text=native_text,
            method="native",
            page=page,
            native_quality=native_score,
            ocr_quality=ocr_score,
        )

    if ocr_text and ocr_score >= ocr_threshold:
        return PreprocessedText(
            text=ocr_text,
            method="ocr",
            page=page,
            native_quality=native_score,
            ocr_quality=ocr_score,
        )

    if native_text and native_score >= ocr_score:
        return PreprocessedText(
            text=native_text,
            method="native_low_quality",
            page=page,
            native_quality=native_score,
            ocr_quality=ocr_score,
        )

    if ocr_text:
        return PreprocessedText(
            text=ocr_text,
            method="ocr_low_quality",
            page=page,
            native_quality=native_score,
            ocr_quality=ocr_score,
        )

    return PreprocessedText(
        text="",
        method="none",
        page=page,
        native_quality=native_score,
        ocr_quality=ocr_score,
    )