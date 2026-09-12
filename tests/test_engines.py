"""Answer-engine selection and the pipeline-assembly fixes. No model is loaded."""

import httpx
import pytest

from generation.llm import FallbackBackend, LLMResponse, OllamaBackend, preferred_ollama_order
from ingestion.documents import Document


# --- Ollama ---------------------------------------------------------------


def test_ollama_context_window_grows_to_fit_and_never_shrinks():
    """Unset, Ollama's small default context silently cut the start of the prompt.
    It never shrinks, because a different num_ctx per request reloads the model."""
    backend = OllamaBackend(num_ctx=4096, client=httpx.Client())
    assert backend._options(1000, 256, 0.0)["num_ctx"] == 4096
    assert backend._options(5000, 256, 0.0)["num_ctx"] == 8192
    assert backend._options(100, 256, 0.0)["num_ctx"] == 8192


def test_preferred_models_come_first():
    assert preferred_ollama_order(["llama3.1:8b", "tinyllama:latest", "qwen2.5:3b"])[0] == "qwen2.5:3b"
    assert preferred_ollama_order(["mistral:7b"]) == ["mistral:7b"]


# --- Groq with an Ollama fallback -----------------------------------------


class _Unreachable:
    name, model = "groq", "llama-3.1-8b-instant"

    def complete(self, prompt, **kwargs):
        raise httpx.ConnectError("no route to host")


class _Local:
    name, model = "ollama", "qwen2.5:3b"

    def complete(self, prompt, **kwargs):
        return LLMResponse(text="Answer [1].", model=self.model, backend=self.name)


def test_fallback_answers_when_the_primary_is_unreachable():
    response = FallbackBackend(_Unreachable(), _Local()).complete("q")
    assert response.backend == "ollama"


def test_fallback_never_splices_a_second_answer_onto_a_streamed_one():
    class _DropsMidStream:
        name, model = "groq", "llama-3.1-8b-instant"

        def complete(self, prompt, *, stream_callback=None, **kwargs):
            stream_callback("Half an answ")
            raise httpx.ReadError("connection reset")

    with pytest.raises(httpx.ReadError):
        FallbackBackend(_DropsMidStream(), _Local()).complete("q", stream_callback=lambda t: None)


# --- ask.py helpers -------------------------------------------------------


def test_ocr_language_is_detected_from_file_names(tmp_path):
    import ask

    rolls = tmp_path / "rolls"
    rolls.mkdir()
    (rolls / "2025-EROLLGEN-S04-183-SIR-FinalRoll-Revision1-HIN-1-WI.pdf").write_bytes(b"%PDF")
    plan = tmp_path / "pmp-2031-report.pdf"
    plan.write_bytes(b"%PDF")
    within = tmp_path / "WITHIN-budget.pdf"
    within.write_bytes(b"%PDF")

    assert ask.detect_ocr_lang(rolls, "auto") == "hin+eng"
    assert ask.detect_ocr_lang(rolls, "en") == "hin+eng"  # the UI's old default
    assert ask.detect_ocr_lang(plan, "auto") == "en"
    assert ask.detect_ocr_lang(plan, "hi") == "hin+eng"
    assert ask.detect_ocr_lang(within, "auto") == "en"


class _FakeDense:
    """Stands in for the vector index; embeds nothing."""

    def __init__(self):
        self.indexed, self.index_calls = [], 0

    def exists(self):
        return bool(self.indexed)

    def count(self):
        return len(self.indexed)

    def recreate(self):
        self.indexed = []

    def index(self, chunks, show_progress=False):
        self.index_calls += 1
        self.indexed = list(chunks)
        return len(self.indexed)

    def search(self, query, limit=10):
        return []

    def get_by_page(self, page_num, limit=50, doc_id=None):
        return []


TABLE_PAGE = Document(
    doc_id="pmp#p66", title="pmp", page=66,
    text="| Sl. No. | Land Use | Proposed (%) |\n| --- | --- | --- |\n"
         "| 1 | Residential | 55.04 |\n| 2 | Commercial | 7.20 |",
)
ROLL_PAGE = Document(
    doc_id="roll#p3", title="roll", page=3,
    text="निर्वाचक नामावली विधानसभा भाग संख्या 12 मतदान केंद्र EPIC SHS1234567 EPIC SHS7654321",
)


def _build(docs, dense_index, tmp_path, monkeypatch):
    import ask

    monkeypatch.setattr(ask, "_MANIFEST_DIR", tmp_path)
    return ask.build_pipeline(docs, "test_collection", backend="stub", use_reranker=False,
                              dense_index=dense_index, progress=lambda message: None)


def test_mixed_folder_keeps_the_master_plans_tables(tmp_path, monkeypatch):
    """One electoral page used to switch every document to the voter-card
    chunker, which dropped the Master Plan's table structure."""
    fake = _FakeDense()
    _build([TABLE_PAGE, ROLL_PAGE], fake, tmp_path, monkeypatch)
    assert any(c.doc_id == "pmp#p66" and c.metadata.get("block_type") == "table"
               for c in fake.indexed)
    assert any(c.doc_id == "roll#p3" for c in fake.indexed)


def test_reranker_follows_the_share_of_each_script():
    """A few quoted Hindi words used to put an English report on the 12-layer
    multilingual reranker, 2-4x slower per question."""
    import ask
    from rerank.cross_encoder import DEFAULT_MODEL, MULTILINGUAL_LIGHT

    english = Document(doc_id="pmp#p1", text="Residential land use is proposed at 55 percent. " * 20
                       + "मास्टर प्लान")
    urdu = Document(doc_id="u#p1", text="یہ اردو کی ایک کتاب ہے " * 20)
    assert ask._choose_reranker([english]) == DEFAULT_MODEL
    assert ask._choose_reranker([ROLL_PAGE]) == MULTILINGUAL_LIGHT
    assert ask._choose_reranker([urdu]) is None


def test_unchanged_corpus_is_not_re_embedded(tmp_path, monkeypatch):
    fake = _FakeDense()
    _build([TABLE_PAGE], fake, tmp_path, monkeypatch)
    _build([TABLE_PAGE], fake, tmp_path, monkeypatch)
    assert fake.index_calls == 1

    edited = Document(doc_id=TABLE_PAGE.doc_id, title="pmp", page=66,
                      text=TABLE_PAGE.text.replace("55.04", "56.00"))
    _build([edited], fake, tmp_path, monkeypatch)
    assert fake.index_calls == 2
