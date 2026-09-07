"""Exact-match deduplication (Phase 1 Core).

Near-duplicate detection is a Stretch item; the hook for it is `fingerprint`,
which is the only thing a MinHash/SimHash implementation would need to replace.
"""

import re
import hashlib
from typing import Iterable, Iterator, Optional

from .documents import Document

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def fingerprint(text: str) -> str:
    """Stable hash of the text's normalised form.

    Normalising before hashing means documents differing only in whitespace,
    case or punctuation collapse together — otherwise near-identical scrapes
    survive dedup and inflate the corpus with redundant evidence.
    """
    normalised = _NON_ALNUM.sub(" ", text.lower()).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def dedup_documents(
    documents: Iterable[Document],
    *,
    protected_ids: Optional[Iterable[str]] = None,
) -> Iterator[Document]:
    """Drop exact/near-exact textual duplicates, first occurrence wins.

    `protected_ids` documents are *never* dropped, and an unprotected duplicate
    of one is removed instead. FiQA contains a group of 38 byte-identical
    documents, one of which is judged by the qrels: without this guarantee that
    judged document can be collapsed away, its query silently loses a relevant
    document, and recall falls — but only at high contamination, where more
    duplicates get sampled. That would look exactly like "contamination hurts
    retrieval", i.e. it would manufacture the project's headline result.
    """
    protected = frozenset(protected_ids or ())
    docs = list(documents)

    # Fingerprints claimed by a protected document. Unprotected duplicates of
    # these are dropped rather than being allowed to displace the protected one.
    protected_fingerprints = {
        fingerprint(d.text) for d in docs if d.doc_id in protected
    }

    seen: set[str] = set()
    for doc in docs:
        fp = fingerprint(doc.text)

        if doc.doc_id in protected:
            # Two protected documents with identical text both survive: dropping
            # either one would corrupt the relevance judgements.
            seen.add(fp)
            yield doc
            continue

        if fp in protected_fingerprints or fp in seen:
            continue

        seen.add(fp)
        yield doc
