"""Tests for the extraction cache and the Indic-script fixes.

None of these need PaddleOCR, PyMuPDF or a model: the cache is exercised
directly, and the script handling is pure text processing.
"""

import json
import os

import pytest

from ingestion.cache import CacheStats, ExtractionCache, file_digest
from ingestion.preprocess import _is_letter, _suspicious_ratio, analyze_native_text
from retrieval.bm25 import tokenize

HINDI = "निर्वाचक नामावली 2025 बिहार कुम्हरार पटना साहिब"


# --- cache --------------------------------------------------------------


@pytest.fixture
def cache(tmp_path):
    return ExtractionCache(root=str(tmp_path / "c"))


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "roll.pdf"
    path.write_bytes(b"%PDF-1.4 fake content")
    return str(path)


def test_miss_then_hit(cache, pdf):
    key = cache.page_key(pdf, 1, {"render_scale": 1.5})
    assert cache.get(key) is None
    cache.put(key, {"text": "निर्वाचक", "metadata": {"ocr_used": True}})
    assert cache.get(key)["text"] == "निर्वाचक"
    assert (cache.stats.hits, cache.stats.misses) == (1, 1)


def test_key_is_content_addressed_not_path_addressed(cache, pdf, tmp_path):
    """A renamed PDF must still hit: OCR is far too expensive to redo."""
    key = cache.page_key(pdf, 1, {})
    moved = str(tmp_path / "renamed.pdf")
    os.rename(pdf, moved)
    assert cache.page_key(moved, 1, {}) == key


def test_edited_file_misses(cache, pdf, tmp_path):
    key = cache.page_key(pdf, 1, {})
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4 DIFFERENT content")
    assert cache.page_key(str(other), 1, {}) != key


def test_pages_do_not_collide(cache, pdf):
    assert cache.page_key(pdf, 1, {}) != cache.page_key(pdf, 2, {})


def test_settings_change_invalidates(cache, pdf):
    """A cache hit must guarantee the text equals a fresh run's output."""
    base = cache.page_key(pdf, 1, {"render_scale": 1.5})
    assert cache.page_key(pdf, 1, {"render_scale": 3.0}) != base
    assert cache.page_key(pdf, 1, {"render_scale": 1.5, "ocr_lang": "hi"}) != base


def test_param_ordering_does_not_change_the_key(cache, pdf):
    a = cache.page_key(pdf, 1, {"a": 1, "b": 2})
    b = cache.page_key(pdf, 1, {"b": 2, "a": 1})
    assert a == b


def test_corrupt_entry_degrades_to_a_miss(cache, pdf):
    """A half-written entry must not abort an hours-long ingest."""
    key = cache.page_key(pdf, 1, {})
    cache.put(key, {"text": "x", "metadata": {}})
    with open(cache._path_for(key), "w", encoding="utf-8") as handle:
        handle.write("{ truncated")
    assert cache.get(key) is None


def test_disabled_cache_never_reads_or_writes(tmp_path, pdf):
    disabled = ExtractionCache(root=str(tmp_path / "d"), enabled=False)
    key = disabled.page_key(pdf, 1, {})
    disabled.put(key, {"text": "x", "metadata": {}})
    assert disabled.get(key) is None
    assert not os.path.exists(tmp_path / "d")


def test_unicode_survives_the_round_trip(cache, pdf):
    key = cache.page_key(pdf, 1, {})
    cache.put(key, {"text": HINDI, "metadata": {}})
    assert cache.get(key)["text"] == HINDI


def test_clear_removes_entries(cache, pdf):
    for page in range(3):
        cache.put(cache.page_key(pdf, page, {}), {"text": "x", "metadata": {}})
    assert cache.clear() == 3


def test_stats_summary():
    stats = CacheStats(hits=28, misses=0, writes=0)
    assert stats.hit_rate == 1.0
    assert "100%" in stats.summary()
    assert CacheStats().hit_rate == 0.0


def test_file_digest_is_stable(pdf):
    assert file_digest(pdf) == file_digest(pdf)
    assert len(file_digest(pdf)) == 64


# --- Devanagari handling ------------------------------------------------


def test_tokenizer_keeps_devanagari_words_whole():
    """`[a-z0-9]+` returned only ['2025']; bare \\w+ split at every matra."""
    assert tokenize(HINDI) == [
        "निर्वाचक", "नामावली", "2025", "बिहार", "कुम्हरार", "पटना", "साहिब",
    ]


def test_tokenizer_still_handles_english_and_mixed_script():
    assert tokenize("The Electoral Roll of Bihar, 2025.") == [
        "electoral", "roll", "bihar", "2025",
    ]
    assert tokenize("Booth 183 - कुम्हरार (सामान्य)") == [
        "booth", "183", "कुम्हरार", "सामान्य",
    ]


def test_combining_marks_count_as_letters():
    assert _is_letter("क") and _is_letter("ा") and _is_letter("्")
    assert not _is_letter(",") and not _is_letter(" ")


def test_devanagari_is_not_scored_as_corrupted():
    """Matras previously read as 'suspicious', forcing OCR on clean Hindi."""
    assert _suspicious_ratio(HINDI) == pytest.approx(0.0)
    report = analyze_native_text(HINDI * 6)
    assert report.suspicious_ratio == pytest.approx(0.0)
    assert report.needs_ocr is False


def test_genuinely_corrupt_text_is_still_flagged():
    assert _suspicious_ratio("���\x01\x02♦♦♦") > 0.5
    assert analyze_native_text("���♦♦♦\x01").needs_ocr is True


# --- Hindi stopwords ----------------------------------------------------


def test_hindi_particles_are_stopped():
    """Unstopped particles made nonsense queries score above real ones.

    "बिल्ली का बच्चा कहाँ सोता है" and "मेरी कार का इंजन खराब है" both scored
    2.295 on the electoral rolls via the shared token "का" — above the genuine
    query "कुम्हरार" (1.560). That destroyed BM25's zero-result abstention.
    """
    assert tokenize("बिल्ली का बच्चा कहाँ सोता है") == ["बिल्ली", "बच्चा", "सोता"]
    assert tokenize("मेरी कार का इंजन खराब है") == ["कार", "इंजन", "खराब"]


def test_domain_content_words_are_not_stopped():
    """The stoplist must not eat the vocabulary the corpus is about."""
    for word in ("मतदान", "केंद्र", "निर्वाचक", "नामावली", "नाम", "संख्या",
                 "क्षेत्र", "कुम्हरार", "विधानसभा"):
        assert tokenize(word) == [word], f"{word} was wrongly stopped"


def test_english_stopwords_still_apply():
    assert tokenize("the roll of the state") == ["roll", "state"]


# --- OCR failure handling -----------------------------------------------


class _BrokenOCR:
    lang = "en"

    def extract_page(self, image_path, page):
        raise RuntimeError("engine crashed")


def _digits_only_pdf(path):
    """A page whose native text is too weak to trust, so OCR is attempted."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "12 34 56")
    doc.save(str(path))
    doc.close()
    return str(path)


def test_ocr_failure_falls_back_to_native_text_without_caching_it(tmp_path):
    """Cached under the page key, the fallback stopped every later run from
    retrying OCR — one transient engine crash became permanent."""
    from ingestion.pdf_extractor import PDFExtractor

    pdf_path = _digits_only_pdf(tmp_path / "scan.pdf")
    cache = ExtractionCache(root=str(tmp_path / "c"))
    docs = list(PDFExtractor(ocr_provider=_BrokenOCR(), cache=cache).extract(pdf_path))

    assert [d.metadata["extraction_method"] for d in docs] == ["native_fallback"]
    assert cache.stats.writes == 0


def test_cache_key_does_not_change_once_the_ocr_engine_is_built(tmp_path):
    """PaddleOCRProvider is an alias, so the class name flipped to
    RobustPaddleOCREngine after the lazy engine was constructed, and a reused
    extractor missed the cache for every file after the first."""
    from ingestion.pdf_extractor import PDFExtractor

    extractor = PDFExtractor(ocr_lang="hi", cache=ExtractionCache(root=str(tmp_path / "c")))
    before = extractor._cache_params()
    extractor._get_ocr_provider()  # lazy engine construction; Paddle itself loads on first page
    assert extractor._cache_params() == before
    assert before["provider"] == "PaddleOCRProvider"


def test_tesseract_page_does_not_report_the_previous_pages_confidence(tmp_path, monkeypatch):
    """Recognition scores from an earlier Paddle page leaked into a later
    Tesseract page and were reported as its confidence."""
    import sys
    import types

    from PIL import Image

    from ingestion.ocr import TesseractOCRProvider

    fake = types.SimpleNamespace(image_to_string=lambda image, lang, config: "Serial 12")
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    image = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(image)

    engine = TesseractOCRProvider(lang="eng")
    engine._last_scores = [0.99]  # left over from a previous Paddle page
    result = engine.extract_page(str(image), page=2)
    assert result.text == "Serial 12"
    assert result.confidence is None
