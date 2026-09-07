"""Text normalisation applied before chunking.

Deliberately conservative: cleaning that rewrites content (lowercasing,
punctuation stripping) belongs in the BM25 tokeniser, not here — the chunk text
is what gets embedded, shown to the user, and cited, so it stays readable.
"""

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SOFT_HYPHEN = "­"
# A newline mid-sentence is almost always PDF line wrapping, not a paragraph.
_WRAPPED_LINE = re.compile(r"(?<=[a-z,;])\n(?=[a-z])")
_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?])")


def clean_text(text: str) -> str:
    """Normalise whitespace, unicode and PDF line-wrapping artefacts."""
    if not text:
        return ""

    # NFKC folds ligatures (ﬁ -> fi) and full-width forms that break tokenising.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_SOFT_HYPHEN, "").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)

    # "inter-\nnational" -> "international", before generic newline collapsing.
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _WRAPPED_LINE.sub(" ", text)

    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)

    return "\n".join(line.strip() for line in text.split("\n")).strip()
