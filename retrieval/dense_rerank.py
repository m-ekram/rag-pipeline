"""Dense retrieval followed by Cross-Encoder reranking."""

from .dense import DenseIndex
from .reranker import CrossEncoderReranker
from .types import ScoredChunk


class DenseRerankRetriever:
    """Dense retrieval followed by Cross-Encoder reranking."""

    def __init__(
        self,
        dense: DenseIndex,
        reranker: CrossEncoderReranker,
        *,
        candidate_limit: int = 50,
    ):
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")

        self.dense = dense
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    @property
    def mode(self) -> str:
        return "dense+rerank"

    def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ScoredChunk]:
        """Retrieve dense candidates and rerank them."""

        if limit <= 0:
            return []

        candidates = self.dense.search(
            query,
            limit=self.candidate_limit,
        )

        return self.reranker.rerank(
            query,
            candidates,
            limit=limit,
        )