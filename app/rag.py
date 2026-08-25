"""The RAG chain: condense -> retrieve -> ground -> answer with citations."""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from app.attribution import attribute
from app.providers import get_chat_model, get_embeddings
from app.retriever import build_retriever, citations, format_context
from app.store import index_meta, load_chunks, load_index

logger = logging.getLogger(__name__)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a documentation assistant. Answer the user's question using ONLY the "
            "numbered passages below.\n"
            "\n"
            "FORMAT REQUIREMENT - this matters more than anything else:\n"
            "Every sentence that states a fact MUST end with the passage number in square "
            "brackets. Small deviations are not acceptable.\n"
            "\n"
            "Correct:\n"
            "  Declare the body with a Pydantic BaseModel subclass [1]. Attributes with a "
            "default value are optional [2].\n"
            "Wrong (no citations - never do this):\n"
            "  Declare the body with a Pydantic BaseModel subclass. Attributes with a "
            "default value are optional.\n"
            "\n"
            "Rules:\n"
            "1. Cite the passage number in square brackets after every claim, like [2]. Cite "
            "several when several support the claim, like [1][3].\n"
            "2. If the passages do not contain the answer, say exactly what is missing and "
            "stop. Never fill the gap from your own knowledge.\n"
            "3. If the passages disagree, say so and cite both.\n"
            "4. Prefer the user's own terminology. Keep it tight: a short paragraph, or a "
            "list when the answer is genuinely a sequence of steps.\n"
            "5. Quote exact values (flags, error codes, limits, commands) verbatim - never "
            "paraphrase an identifier.\n"
            "\n"
            "Passages:\n"
            "{context}",
        ),
        ("human", "{question}"),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the follow-up question as a standalone search query that makes sense "
            "without the conversation. Resolve every pronoun and implicit reference against "
            "the history. Output the query only - no preamble, no quotes.\n"
            "\n"
            "Conversation so far:\n"
            "{history}",
        ),
        ("human", "Follow-up: {question}"),
    ]
)

NO_CONTEXT_MESSAGE = (
    "I could not find anything in the indexed documents that addresses that. "
    "Either the topic is not covered, or the wording is far enough from the source "
    "text that retrieval missed it - try naming the specific feature, error code, or file."
)


def _format_history(history: list[dict]) -> str:
    turns = history[-config.MAX_HISTORY_TURNS :]
    return "\n".join(f"{t.get('role', 'user').capitalize()}: {t.get('content', '')}" for t in turns)


class RagEngine:
    """Holds the loaded index and retriever. Build once, reuse per request."""

    def __init__(self, index_dir: str | None = None):
        self.store = load_index(index_dir)
        self.chunks = load_chunks(index_dir)
        self.retriever = build_retriever(self.store, self.chunks)
        self.meta = index_meta(index_dir)
        self.llm = get_chat_model()
        self._embeddings = get_embeddings()
        logger.info("RAG engine ready | %s | %d chunks", config.summary(), len(self.chunks))

    # -- pipeline steps --------------------------------------------------------

    def condense(self, question: str, history: list[dict] | None) -> str:
        if not history:
            return question
        chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        try:
            rewritten = chain.invoke({"history": _format_history(history), "question": question}).strip()
            return rewritten or question
        except Exception as exc:  # never let query rewriting break the answer
            logger.warning("Condense step failed (%s); using the raw question", exc)
            return question

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        retriever = self.retriever if top_k is None else build_retriever(self.store, self.chunks, top_k)
        return retriever.invoke(query)[: top_k or config.TOP_K]

    # -- public API ------------------------------------------------------------

    def add_citations(self, answer: str, docs: list[Document]) -> str:
        """Fill in [n] markers when the model did not emit any itself."""
        if config.CITATION_MODE != "auto":
            return answer
        return attribute(answer, docs, self._embeddings)

    def ask(self, question: str, history: list[dict] | None = None, top_k: int | None = None) -> dict:
        started = time.time()
        query = self.condense(question, history)
        docs = self.retrieve(query, top_k)

        if not docs:
            return {
                "answer": NO_CONTEXT_MESSAGE,
                "sources": [],
                "search_query": query,
                "latency_ms": int((time.time() - started) * 1000),
            }

        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"context": format_context(docs), "question": question})
        answer = self.add_citations(answer, docs)

        return {
            "answer": answer.strip(),
            "sources": citations(docs),
            "search_query": query,
            "latency_ms": int((time.time() - started) * 1000),
        }

    async def astream(
        self, question: str, history: list[dict] | None = None, top_k: int | None = None
    ) -> AsyncIterator[dict]:
        """Yield {'type': 'sources'|'token'|'done'|'error', ...} events for SSE."""
        try:
            query = self.condense(question, history)
            docs = self.retrieve(query, top_k)
            yield {"type": "sources", "search_query": query, "sources": citations(docs)}

            if not docs:
                yield {"type": "token", "text": NO_CONTEXT_MESSAGE}
                yield {"type": "done"}
                return

            chain = ANSWER_PROMPT | self.llm | StrOutputParser()
            streamed = []
            async for piece in chain.astream({"context": format_context(docs), "question": question}):
                if piece:
                    streamed.append(piece)
                    yield {"type": "token", "text": piece}

            # Attribution needs the finished text, so the annotated version is
            # sent as a final replacement rather than mid-stream.
            raw = "".join(streamed)
            cited = self.add_citations(raw, docs)
            if cited != raw:
                yield {"type": "replace", "text": cited}
            yield {"type": "done"}
        except Exception as exc:
            logger.exception("Streaming failed")
            yield {"type": "error", "message": str(exc)}
