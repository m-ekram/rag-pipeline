"""Corpus assembly: documents in, chunks out, with the contamination knobs
the project's goal depends on.

`build_corpus` is the single place that decides *which* documents exist for a
given run. The contamination sweep (plan: "The unique deliverable") works by
calling it with different distractor counts and nothing else changing.
"""

import random
import logging
from typing import Iterable, Iterator, Optional

from .chunking import Chunker, FixedSizeChunker
from .dedup import dedup_documents
from .documents import Chunk, Document
from .loaders import load_beir_corpus, load_noisy_corpus, load_qrels

logger = logging.getLogger(__name__)


def judged_doc_ids(qrels_path: Optional[str] = None) -> set[str]:
    """Every doc id appearing in the qrels — these must never be sampled out."""
    qrels = load_qrels(qrels_path)
    return {doc_id for judgements in qrels.values() for doc_id in judgements}


def build_corpus(
    *,
    corpus_path: Optional[str] = None,
    qrels_path: Optional[str] = None,
    in_domain_distractors: Optional[int] = None,
    noisy_corpus_path: Optional[str] = None,
    out_of_domain_distractors: int = 0,
    seed: int = 13,
    dedup: bool = True,
) -> list[Document]:
    """Assemble a corpus at a chosen contamination level.

    All judged documents are always included — dropping one would silently make
    a query unanswerable and corrupt recall. Contamination is controlled purely
    by how many *unjudged* documents get added on top.

    `in_domain_distractors=None` means "use every unjudged document" (the full
    57,638-doc FiQA corpus, ~0.97 contamination). `0` means judged docs only.
    """
    judged = judged_doc_ids(qrels_path)
    rng = random.Random(seed)

    kept: list[Document] = []
    unjudged: list[Document] = []
    for doc in load_beir_corpus(corpus_path):
        (kept if doc.doc_id in judged else unjudged).append(doc)

    logger.info("Loaded %d judged and %d unjudged documents",
                len(kept), len(unjudged))

    if in_domain_distractors is None:
        kept.extend(unjudged)
    elif in_domain_distractors > 0:
        n = min(in_domain_distractors, len(unjudged))
        if n < in_domain_distractors:
            logger.warning("Only %d unjudged docs available, wanted %d",
                           n, in_domain_distractors)
        kept.extend(rng.sample(unjudged, n))

    if out_of_domain_distractors:
        noise = list(load_noisy_corpus(noisy_corpus_path))
        n = min(out_of_domain_distractors, len(noise))
        if n < out_of_domain_distractors:
            logger.warning(
                "Noisy corpus holds only %d docs, wanted %d — the out-of-domain "
                "contamination axis will be weaker than planned.", n,
                out_of_domain_distractors,
            )
        kept.extend(rng.sample(noise, n))

    if dedup:
        before = len(kept)
        # Judged documents are protected: a duplicate must never displace one.
        kept = list(dedup_documents(kept, protected_ids=judged))
        if before != len(kept):
            logger.info("Dedup removed %d documents", before - len(kept))

    # Post-condition, not an optimism check: every contamination level in the
    # sweep must hold the judged set fixed, or the curves compare corpora that
    # differ in relevant documents as well as in distractors.
    surviving = {d.doc_id for d in kept} & judged
    if len(surviving) != len(judged):
        raise RuntimeError(
            f"Corpus assembly lost {len(judged) - len(surviving)} judged "
            f"document(s): {sorted(judged - surviving)[:10]}"
        )

    rng.shuffle(kept)
    logger.info("Corpus assembled: %d documents (contamination %.4f)",
                len(kept), 1 - (len(judged) / len(kept)) if kept else 0.0)
    return kept


def chunk_corpus(
    documents: Iterable[Document], chunker: Optional[Chunker] = None
) -> Iterator[Chunk]:
    chunker = chunker or FixedSizeChunker()
    for doc in documents:
        yield from chunker.chunk(doc)


def contamination_ratio(documents: Iterable[Document],
                        judged: Optional[set[str]] = None) -> float:
    """Fraction of the corpus that cannot answer any judged query."""
    docs = list(documents)
    if not docs:
        return 0.0
    judged = judged if judged is not None else judged_doc_ids()
    return 1.0 - sum(1 for d in docs if d.doc_id in judged) / len(docs)
