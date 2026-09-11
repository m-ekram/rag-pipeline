"""Adaptive dual PDF extraction.

Pipeline:
1. Try native PDF text extraction first.
2. Measure native text quality.
3. Use native text when it is good enough.
4. Initialize OCR only when native extraction is insufficient.
5. Reuse the OCR provider for later pages that need OCR.

This keeps OCR out of the critical path for normal text PDFs.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import tempfile
import time
from typing import Optional

import pymupdf
from pypdf import PdfReader

from .cache import ExtractionCache
from .clean import clean_text
from .documents import Document
from .layout import DigitalLayoutExtractor, stitch_continuation_tables
from .ocr import OCRProvider, PaddleOCRProvider, OCRResult
from .preprocess import ExtractionResult, reconcile_extractions, text_quality, PreprocessedText

_WORKER_OCR: Optional[PaddleOCRProvider] = None


def _ocr_worker_init(lang: str, text_det_unclip_ratio: float = 1.8, provider_name: str = "PaddleOCRProvider"):
    """Initializer for background OCR worker processes (1 core per worker, reserving 1 core for SSH/system)."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PADDLE_NUM_THREADS"] = "1"

    # Restrict worker processes to non-last CPU cores, reserving 1 core for SSH & system
    if hasattr(os, "sched_setaffinity"):
        try:
            total_cpus = os.cpu_count() or 1
            if total_cpus > 1:
                # Cores 0 .. total_cpus - 2 used by workers; core total_cpus - 1 kept free
                os.sched_setaffinity(0, set(range(total_cpus - 1)))
        except Exception:
            pass

    from .ocr import PaddleOCRProvider, TesseractOCRProvider
    global _WORKER_OCR
    if provider_name == "TesseractOCRProvider":
        _WORKER_OCR = TesseractOCRProvider(lang=lang, text_det_unclip_ratio=text_det_unclip_ratio)
    else:
        _WORKER_OCR = PaddleOCRProvider(lang=lang, text_det_unclip_ratio=text_det_unclip_ratio)


def _ocr_page_worker_task(pdf_path_str: str, page_number: int, render_scale: float) -> tuple[int, str, float | None, str]:
    """Execute single page OCR inside a background worker process."""
    global _WORKER_OCR
    import pymupdf
    pdf = pymupdf.open(pdf_path_str)
    try:
        pdf_page = pdf[page_number - 1]
        pixmap = pdf_page.get_pixmap(
            dpi=150,
            alpha=False,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            image_path = Path(temp.name)
        try:
            pixmap.save(str(image_path))
            result = _WORKER_OCR.extract_page(str(image_path), page_number)
            return page_number, result.text, result.confidence, result.engine
        finally:
            image_path.unlink(missing_ok=True)
    finally:
        pdf.close()


class PDFExtractor:
    """General-purpose adaptive PDF extractor."""

    def __init__(
        self,
        ocr_provider: OCRProvider | None = None,
        ocr_lang: str = "en",
        render_scale: float = 1.5,
        text_det_unclip_ratio: float = 1.8,
        native_threshold: float = 0.55,
        ocr_threshold: float = 0.30,
        cache: ExtractionCache | None = None,
        use_cache: bool = True,
        workers: int = 1,
    ):
        # Store the provider, but DO NOT create PaddleOCR here.
        self._ocr_provider = ocr_provider
        # Canonicalize language aliases so cache keys match across hin+eng, hi, hin
        normalized_lang = (ocr_lang or "en").lower().strip()
        if "urd" in normalized_lang or "urdu" in normalized_lang:
            self.ocr_lang = "urd"
        elif "hin" in normalized_lang or "hi" in normalized_lang:
            self.ocr_lang = "hin+eng"
        else:
            self.ocr_lang = ocr_lang

        self.render_scale = render_scale
        self.text_det_unclip_ratio = text_det_unclip_ratio
        self.native_threshold = native_threshold
        self.ocr_threshold = ocr_threshold
        self.workers = workers
        self.digital_layout = DigitalLayoutExtractor()

        # Page results are cached by (file content, page, settings). A cache hit
        # also means PaddleOCR is never constructed, which saves ~5s of startup
        # on top of ~6s per page.
        self.cache = cache if cache is not None else ExtractionCache(enabled=use_cache)

    def _cache_params(self) -> dict:
        """Everything that changes extraction output, folded into the key."""
        return {
            "render_scale": self.render_scale,
            "text_det_unclip_ratio": self.text_det_unclip_ratio,
            "native_threshold": self.native_threshold,
            "ocr_threshold": self.ocr_threshold,
            "ocr_lang": self.ocr_lang,
            "provider": type(self._ocr_provider).__name__
            if self._ocr_provider is not None
            else "PaddleOCRProvider",
        }

    def _get_ocr_provider(self) -> OCRProvider:
        """Create OCR provider only when OCR is actually required."""
        if self._ocr_provider is None:
            self._ocr_provider = PaddleOCRProvider(
                lang=self.ocr_lang, text_det_unclip_ratio=self.text_det_unclip_ratio
            )

        return self._ocr_provider

    def _ocr_page(
        self,
        pdf_page,
        page_number: int,
    ) -> OCRResult:
        """Render one PDF page and run OCR on it."""

        pixmap = pdf_page.get_pixmap(
            dpi=150,
            alpha=False,
        )

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as temp:
            image_path = Path(temp.name)

        try:
            pixmap.save(str(image_path))

            provider = self._get_ocr_provider()

            return provider.extract_page(
                str(image_path),
                page_number,
            )

        finally:
            image_path.unlink(missing_ok=True)

    def extract(
        self,
        path: str,
        *,
        title: str | None = None,
        clean: bool = True,
        max_pages: int | None = None,
    ):
        """Extract Documents from every PDF page adaptively."""

        pdf_path = Path(path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        reader = PdfReader(str(pdf_path))
        pdf = pymupdf.open(str(pdf_path))

        doc_id = pdf_path.stem
        document_title = title or doc_id

        # Urdu InPage / Nastaliq font auto-detection
        # Bypasses the broken InPage digital text stream completely
        # and forces 150 DPI Tesseract Urdu OCR
        is_urdu_doc = any(k in str(pdf_path).lower() for k in ["urdu", "bang-i-dara", "iqbal"]) or "urd" in self.ocr_lang.lower()
        force_ocr = is_urdu_doc
        if is_urdu_doc:
            self.ocr_lang = "urd"
            if self._ocr_provider is not None and getattr(self._ocr_provider, "lang", "") != "urd":
                self._ocr_provider = None

        total_pages = len(reader.pages)
        if max_pages is not None:
            total_pages = min(total_pages, max_pages)

        page_docs: dict[int, Document] = {}
        pages_needing_ocr: list[tuple[int, ExtractionResult, float, str]] = []

        try:
            params = self._cache_params()

            # --------------------------------------------------
            # PASS 1: Fast Scan (Cache hits + High-quality native text)
            # --------------------------------------------------
            for page_number in range(1, total_pages + 1):
                page = reader.pages[page_number - 1]
                cache_key = self.cache.page_key(str(pdf_path), page_number, params)
                cached = self.cache.get(cache_key)

                if cached is not None:
                    # Invalidate if it was native extraction for an Urdu document (bypasses corrupted InPage font)
                    if force_ocr and cached.get("metadata", {}).get("extraction_method", "").startswith("native"):
                        cached = None
                    else:
                        print(f"  [+] Page {page_number}/{total_pages}: loaded from cache.", flush=True)
                        text = clean_text(cached["text"]) if clean else cached["text"]
                        if text.strip():
                            page_docs[page_number] = Document(
                                doc_id=f"{doc_id}#p{page_number}",
                                text=text,
                                title=document_title,
                                source=str(pdf_path),
                                page=page_number,
                                metadata={**cached["metadata"], "from_cache": True},
                            )
                        continue

                native_text = page.extract_text() or ""
                native_result = ExtractionResult(
                    text=native_text,
                    method="native",
                    page=page_number,
                )
                native_quality = text_quality(native_text)

                if not force_ocr and native_text.strip() and native_quality >= self.native_threshold:
                    print(f"  [+] Page {page_number}/{total_pages}: native text good (quality: {native_quality:.2f}).", flush=True)
                    has_table = False
                    try:
                        layout_blocks = self.digital_layout.extract_page(pdf[page_number - 1], page_number)
                        has_table = any(b.type == "table" for b in layout_blocks)
                        if has_table:
                            final_text = "\n\n".join(b.content for b in layout_blocks if b.type != "footer")
                        else:
                            final_text = native_text
                    except Exception:
                        final_text = native_text

                    # Low-Density Diagram Recovery (e.g. PRISMA flowcharts, vector diagrams)
                    fitz_page = pdf[page_number - 1]
                    is_diagram = (
                        len(native_text.strip()) < 400
                        and (len(fitz_page.get_drawings()) > 5 or len(fitz_page.get_images()) > 0)
                    )
                    diag_recovered = False
                    if is_diagram:
                        try:
                            diag_res = self._ocr_page(fitz_page, page_number)
                            if diag_res.text.strip():
                                final_text += f"\n\n[Diagram / Vector Box Content]:\n{diag_res.text.strip()}"
                                diag_recovered = True
                        except Exception as d_err:
                            logger.debug("Diagram OCR recovery note: %s", d_err)

                    metadata = {
                        "extraction_method": "native_layout" if has_table else "native",
                        "has_tables": has_table,
                        "has_diagram": is_diagram,
                        "diagram_recovered": diag_recovered,
                        "native_quality": native_quality,
                        "ocr_quality": 0.0,
                        "ocr_used": diag_recovered,
                        "ocr_engine": None,
                        "ocr_confidence": None,
                    }
                    self.cache.put(cache_key, {"text": final_text, "metadata": metadata})
                    text = clean_text(final_text) if clean else final_text
                    if text.strip():
                        page_docs[page_number] = Document(
                            doc_id=f"{doc_id}#p{page_number}",
                            text=text,
                            title=document_title,
                            source=str(pdf_path),
                            page=page_number,
                            metadata={**metadata, "from_cache": False},
                        )
                else:
                    pages_needing_ocr.append((page_number, native_result, native_quality, cache_key))

            # --------------------------------------------------
            # PASS 2: OCR Execution (Sequential or Parallel)
            # --------------------------------------------------
            if pages_needing_ocr:
                num_to_ocr = len(pages_needing_ocr)
                if num_to_ocr <= 2 or self.workers <= 1 or sys.platform == "win32":
                    for page_number, native_result, native_quality, cache_key in pages_needing_ocr:
                        print(f"  [*] Page {page_number}/{total_pages}: running OCR...", flush=True)
                        t0 = time.perf_counter()
                        try:
                            ocr_result = self._ocr_page(pdf[page_number - 1], page_number)
                            if force_ocr:
                                final = PreprocessedText(
                                    text=ocr_result.text,
                                    method="ocr",
                                    page=page_number,
                                    native_quality=0.0,
                                    ocr_quality=text_quality(ocr_result.text),
                                )
                                metadata = {
                                    "extraction_method": "ocr",
                                    "native_quality": 0.0,
                                    "ocr_quality": final.ocr_quality,
                                    "ocr_used": True,
                                    "ocr_engine": ocr_result.engine,
                                    "ocr_confidence": ocr_result.confidence,
                                }
                            else:
                                final = reconcile_extractions(
                                    native_result,
                                    ExtractionResult(text=ocr_result.text, method="ocr", page=page_number, confidence=ocr_result.confidence),
                                    native_threshold=self.native_threshold,
                                    ocr_threshold=self.ocr_threshold,
                                )
                                metadata = {
                                    "extraction_method": final.method,
                                    "native_quality": native_quality,
                                    "ocr_quality": final.ocr_quality,
                                    "ocr_used": True,
                                    "ocr_engine": ocr_result.engine,
                                    "ocr_confidence": ocr_result.confidence,
                                }
                            conf_str = f"conf: {ocr_result.confidence:.2f}" if ocr_result.confidence is not None else ""
                        except Exception as ocr_err:
                            print(f"  [!] OCR failed on page {page_number} ({ocr_err}). Falling back to native text.", flush=True)
                            final = native_result
                            metadata = {
                                "extraction_method": "native_fallback",
                                "native_quality": native_quality,
                                "ocr_quality": 0.0,
                                "ocr_used": False,
                                "ocr_engine": None,
                                "ocr_confidence": None,
                            }
                            conf_str = "native fallback"

                        elapsed_page = time.perf_counter() - t0
                        self.cache.put(cache_key, {"text": final.text, "metadata": metadata})
                        print(f"  [+] Page {page_number}/{total_pages}: OCR completed in {elapsed_page:.1f}s ({len(final.text)} chars, {conf_str}).", flush=True)
                        text = clean_text(final.text) if clean else final.text
                        if text.strip():
                            page_docs[page_number] = Document(
                                doc_id=f"{doc_id}#p{page_number}",
                                text=text,
                                title=document_title,
                                source=str(pdf_path),
                                page=page_number,
                                metadata={**metadata, "from_cache": False},
                            )
                else:
                    active_workers = min(self.workers, num_to_ocr)
                    print(f"[*] Processing {num_to_ocr} pages in parallel with {active_workers} worker processes...", flush=True)
                    t_start_pool = time.perf_counter()
                    provider_cls_name = type(self._ocr_provider).__name__ if self._ocr_provider is not None else "PaddleOCRProvider"
                    try:
                        with ProcessPoolExecutor(max_workers=active_workers, initializer=_ocr_worker_init, initargs=(self.ocr_lang, self.text_det_unclip_ratio, provider_cls_name)) as pool:
                            futures = {
                                pool.submit(_ocr_page_worker_task, str(pdf_path), p[0], self.render_scale): p[0]
                                for p in pages_needing_ocr
                            }
                            for fut in as_completed(futures):
                                page_num, ocr_text, ocr_conf, ocr_eng = fut.result()
                                _, native_result, native_quality, cache_key = ocr_lookup[page_num]
                                if force_ocr:
                                    final = PreprocessedText(
                                        text=ocr_text,
                                        method="ocr",
                                        page=page_num,
                                        native_quality=0.0,
                                        ocr_quality=text_quality(ocr_text),
                                    )
                                    metadata = {
                                        "extraction_method": "ocr",
                                        "native_quality": 0.0,
                                        "ocr_quality": final.ocr_quality,
                                        "ocr_used": True,
                                        "ocr_engine": ocr_eng,
                                        "ocr_confidence": ocr_conf,
                                    }
                                else:
                                    final = reconcile_extractions(
                                        native_result,
                                        ExtractionResult(text=ocr_text, method="ocr", page=page_num, confidence=ocr_conf),
                                        native_threshold=self.native_threshold,
                                        ocr_threshold=self.ocr_threshold,
                                    )
                                    metadata = {
                                        "extraction_method": final.method,
                                        "native_quality": native_quality,
                                        "ocr_quality": final.ocr_quality,
                                        "ocr_used": True,
                                        "ocr_engine": ocr_eng,
                                        "ocr_confidence": ocr_conf,
                                    }
                                self.cache.put(cache_key, {"text": final.text, "metadata": metadata})
                                conf_str = f"conf: {ocr_conf:.2f}" if ocr_conf is not None else ""
                                print(f"  [+] Page {page_num}/{total_pages}: OCR completed ({len(final.text)} chars, {conf_str}).", flush=True)
                                text = clean_text(final.text) if clean else final.text
                                if text.strip():
                                    page_docs[page_num] = Document(
                                        doc_id=f"{doc_id}#p{page_num}",
                                        text=text,
                                        title=document_title,
                                        source=str(pdf_path),
                                        page=page_num,
                                        metadata={**metadata, "from_cache": False},
                                    )
                        pool_elapsed = time.perf_counter() - t_start_pool
                        print(f"[+] Finished parallel OCR of {num_to_ocr} pages in {pool_elapsed:.1f}s (avg {pool_elapsed/num_to_ocr:.1f}s/page).", flush=True)
                    except Exception as pool_err:
                        print(f"  [!] Worker pool failed ({pool_err}). Falling back to sequential processing...", flush=True)
                        for page_number, native_result, native_quality, cache_key in pages_needing_ocr:
                            if page_number in page_docs:
                                continue
                            self.cache.put(cache_key, {"text": native_result.text, "metadata": {"extraction_method": "native_fallback"}})
                            text = clean_text(native_result.text) if clean else native_result.text
                            if text.strip():
                                page_docs[page_number] = Document(
                                    doc_id=f"{doc_id}#p{page_number}",
                                    text=text,
                                    title=document_title,
                                    source=str(pdf_path),
                                    page=page_number,
                                    metadata={"extraction_method": "native_fallback", "from_cache": False},
                                )

            # --------------------------------------------------
            # PASS 3: Multi-Page Table Stitching & Yield in strict page order
            # --------------------------------------------------
            ordered_docs = [page_docs[p] for p in range(1, total_pages + 1) if p in page_docs]
            stitched_docs = stitch_continuation_tables(ordered_docs)
            for doc in stitched_docs:
                yield doc

        finally:
            pdf.close()