"""Loading: text normalisation and the metadata every citation depends on.

If `source` is wrong or missing, every downstream citation points at the wrong
file, so the metadata assertions here matter more than they look.
"""

import pytest

from app.loaders import SUPPORTED_SUFFIXES, clean_text, load_directory, load_file


# -- clean_text ----------------------------------------------------------------


def test_rejoins_words_broken_across_lines():
    """PDF extraction hyphenates at line ends; left alone it wrecks BM25."""
    assert clean_text("config-\nuration") == "configuration"


def test_collapses_runs_of_spaces_and_tabs():
    assert clean_text("a  \t  b") == "a b"


def test_collapses_non_breaking_spaces():
    assert clean_text("a\u00a0\u00a0b") == "a b"


def test_caps_blank_lines_at_two():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_keeps_a_single_paragraph_break():
    assert clean_text("a\n\nb") == "a\n\nb"


def test_normalises_windows_and_classic_mac_line_endings():
    assert clean_text("a\r\nb") == "a\nb"
    assert clean_text("a\rb") == "a\nb"


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("\n\n  hello  \n\n") == "hello"


def test_empty_input_stays_empty():
    assert clean_text("") == ""


# -- per-format loading --------------------------------------------------------


def test_markdown_carries_source_and_title(tmp_path):
    (tmp_path / "networking-guide.md").write_text("# Guide\n\nThe timeout is 30s.")
    docs = load_file(tmp_path / "networking-guide.md", tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "networking-guide.md"
    assert docs[0].metadata["title"] == "networking guide"
    assert "timeout" in docs[0].page_content


def test_source_is_posix_relative_even_in_subdirectories(tmp_path):
    nested = tmp_path / "manuals" / "net"
    nested.mkdir(parents=True)
    (nested / "a.md").write_text("The timeout is 30 seconds and that matters.")
    docs = load_file(nested / "a.md", tmp_path)
    assert docs[0].metadata["source"] == "manuals/net/a.md"


def test_html_drops_script_style_nav_and_footer(tmp_path):
    (tmp_path / "page.html").write_text(
        "<html><head><title>Real Title</title><style>.x{}</style></head>"
        "<body><nav>MENU</nav><p>Body text here.</p>"
        "<script>alert(1)</script><footer>FOOT</footer></body></html>"
    )
    docs = load_file(tmp_path / "page.html", tmp_path)
    text = docs[0].page_content
    assert "Body text here." in text
    assert "MENU" not in text and "FOOT" not in text
    assert "alert" not in text and ".x{}" not in text
    assert docs[0].metadata["title"] == "Real Title"


def test_plain_text_and_rst_load(tmp_path):
    (tmp_path / "notes.txt").write_text("Plain content that is long enough to keep.")
    (tmp_path / "spec.rst").write_text("RST content that is long enough to keep.")
    assert load_file(tmp_path / "notes.txt", tmp_path)[0].page_content.startswith("Plain")
    assert load_file(tmp_path / "spec.rst", tmp_path)[0].page_content.startswith("RST")


def test_an_empty_file_yields_no_documents(tmp_path):
    (tmp_path / "empty.md").write_text("   \n\n  ")
    assert load_file(tmp_path / "empty.md", tmp_path) == []


# -- directory walk ------------------------------------------------------------


def test_load_directory_skips_unsupported_suffixes(tmp_path):
    (tmp_path / "keep.md").write_text("Supported content, long enough to survive.")
    (tmp_path / "skip.xlsx").write_text("nope")
    (tmp_path / "skip.png").write_text("nope")
    docs = load_directory(tmp_path)
    assert [d.metadata["source"] for d in docs] == ["keep.md"]


def test_load_directory_recurses(tmp_path):
    (tmp_path / "top.md").write_text("Top level content, long enough to survive.")
    sub = tmp_path / "deep" / "deeper"
    sub.mkdir(parents=True)
    (sub / "low.md").write_text("Nested content, long enough to survive.")
    sources = {d.metadata["source"] for d in load_directory(tmp_path)}
    assert sources == {"top.md", "deep/deeper/low.md"}


def test_load_directory_is_deterministically_ordered(tmp_path):
    for name in ("c.md", "a.md", "b.md"):
        (tmp_path / name).write_text(f"Content for {name}, long enough to survive.")
    sources = [d.metadata["source"] for d in load_directory(tmp_path)]
    assert sources == sorted(sources)


def test_a_missing_directory_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="Data directory not found"):
        load_directory(tmp_path / "nope")


def test_a_directory_with_nothing_supported_fails_loudly(tmp_path):
    (tmp_path / "only.xlsx").write_text("nope")
    with pytest.raises(SystemExit, match="No supported documents"):
        load_directory(tmp_path)


def test_one_unreadable_file_does_not_kill_the_run(tmp_path, monkeypatch):
    """A corrupt file in a 141-file corpus should cost that file, not the run."""
    (tmp_path / "good.md").write_text("Good content, long enough to survive.")
    (tmp_path / "bad.pdf").write_bytes(b"not really a pdf")
    docs = load_directory(tmp_path)
    assert [d.metadata["source"] for d in docs] == ["good.md"]


def test_the_supported_set_is_what_the_readme_claims():
    assert SUPPORTED_SUFFIXES == {
        ".pdf", ".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".docx"
    }
