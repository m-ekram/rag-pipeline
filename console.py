"""Console encoding guard, importable from any entry point.

On Windows the console still defaults to a legacy code page (cp1252 / cp437) on
Python 3.11, so printing Devanagari or Urdu raises:

    UnicodeEncodeError: 'charmap' codec can't encode characters in position ...

That kills a script *after* the expensive OCR and retrieval work has already
completed — the worst possible time. macOS and Linux are UTF-8 already, so this
is a no-op there.

`errors="replace"` is deliberate: a mangled glyph in a progress line is a far
better outcome than losing a completed run to an encoding error.
"""

import sys


def use_utf8_console(errors: str = "replace") -> bool:
    """Reconfigure stdout/stderr to UTF-8. Returns True if anything changed."""
    changed = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # redirected to something without a text-stream API
        current = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if current == "utf8" and getattr(stream, "errors", None) == errors:
            continue
        try:
            reconfigure(encoding="utf-8", errors=errors)
            changed = True
        except (ValueError, OSError):
            # Detached or non-reconfigurable stream; printing may still fail,
            # but that is strictly no worse than before.
            pass
    return changed


def safe(text: str) -> str:
    """Force text through the console's encoding, replacing what won't fit."""
    encoding = (getattr(sys.stdout, "encoding", None) or "utf-8")
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
