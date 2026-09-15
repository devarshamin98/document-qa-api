"""Orchestration: ingest once, retrieve per question, answer concurrently."""

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import structlog

from app.config import Settings
from app.core.ports import Citation, EmbeddingsClient, LLMClient, ScoredChunk, VectorStore
from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import EmptyDocumentError, Page
from app.qa.prompts import NOT_FOUND, SYSTEM_PROMPT, build_user_prompt, is_not_found, parse_answer

log = structlog.get_logger(__name__)

EXCERPT_CHARS = 240
# Questions can carry customer specifics, so logs get a prefix, never the whole
# thing — and never any document text.
QUESTION_LOG_CHARS = 80


@dataclass(frozen=True, slots=True)
class Answer:
    """One question's outcome.

    Three states, because a failure and a correct "the document doesn't say" are
    opposite facts: answered (`found`), genuinely absent (`not found`, no error),
    and failed (`error` set).
    """

    question: str
    answer: str
    found: bool
    error: str | None = None
    citations: list[Citation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QAResult:
    answers: list[Answer]
    chunk_count: int
    embedding_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class QAService:
    """Answers a batch of questions against one document."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        embeddings: EmbeddingsClient,
        store_factory: Callable[[], VectorStore],
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._store_factory = store_factory
        self._settings = settings

    async def answer(self, *, pages: Sequence[Page], questions: Sequence[str]) -> QAResult:
        """Chunk, embed and index the document once, then answer every question."""
        started = time.perf_counter()

        chunks = chunk_pages(
            pages,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise EmptyDocumentError("The document produced no indexable text.")

        # One embedding pass over the document, one over all the questions.
        # Never per question — that is the wasted-LLM-call trap.
        doc_embedded = await self._embeddings.embed_texts([chunk.text for chunk in chunks])
        store = self._store_factory()
        store.add(chunks, doc_embedded.vectors)

        question_embedded = await self._embeddings.embed_texts(list(questions))

        semaphore = asyncio.Semaphore(self._settings.max_concurrent_llm_calls)
        outcomes = await asyncio.gather(
            *(
                self._answer_one(store, semaphore, index, question, vector)
                for index, (question, vector) in enumerate(
                    zip(questions, question_embedded.vectors, strict=True)
                )
            )
        )

        answers = [answer for answer, _, _ in outcomes]
        latency_ms = int((time.perf_counter() - started) * 1000)

        embedding_tokens = doc_embedded.tokens + question_embedded.tokens
        prompt_tokens = sum(prompt for _, prompt, _ in outcomes)
        completion_tokens = sum(completion for _, _, completion in outcomes)

        log.info(
            "qa_completed",
            questions=len(answers),
            chunks=len(chunks),
            found=sum(1 for answer in answers if answer.found),
            errors=sum(1 for answer in answers if answer.error),
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            tokens_embedding=embedding_tokens,
            latency_ms=latency_ms,
        )

        return QAResult(
            answers=answers,
            chunk_count=len(chunks),
            embedding_tokens=embedding_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    async def _answer_one(
        self,
        store: VectorStore,
        semaphore: asyncio.Semaphore,
        index: int,
        question: str,
        vector: Sequence[float],
    ) -> tuple[Answer, int, int]:
        started = time.perf_counter()
        hits = await store.search(vector, self._settings.top_k)

        if not hits or hits[0].score < self._settings.min_score:
            # Nothing retrieved clears the floor, so there is no point spending a
            # completion to be told so.
            log.info(
                "question_answered",
                question_index=index,
                question=question[:QUESTION_LOG_CHARS],
                found=False,
                reason="below_score_floor",
                top_score=round(hits[0].score, 4) if hits else None,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            return Answer(question=question, answer=NOT_FOUND, found=False), 0, 0

        prompt = build_user_prompt(question, [hit.chunk for hit in hits])
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    self._llm.complete(SYSTEM_PROMPT, prompt),
                    timeout=self._settings.llm_timeout_s,
                )
            except TimeoutError:
                log.warning(
                    "question_timeout",
                    question_index=index,
                    question=question[:QUESTION_LOG_CHARS],
                    timeout_s=self._settings.llm_timeout_s,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                return (
                    Answer(
                        question=question,
                        answer=NOT_FOUND,
                        found=False,
                        error="TIMEOUT",
                    ),
                    0,
                    0,
                )

        text, cited_positions = parse_answer(result.text)
        found = not is_not_found(text)
        citations = _build_citations(hits, cited_positions) if found else []

        log.info(
            "question_answered",
            question_index=index,
            question=question[:QUESTION_LOG_CHARS],
            found=found,
            citations=len(citations),
            top_score=round(hits[0].score, 4),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

        return (
            Answer(
                question=question,
                answer=text if found else NOT_FOUND,
                found=found,
                citations=citations,
            ),
            result.prompt_tokens,
            result.completion_tokens,
        )


def _build_citations(hits: Sequence[ScoredChunk], cited: Sequence[int]) -> list[Citation]:
    """Map the passage positions the model cited back onto the retrieved chunks.

    `cited` holds 1-based positions in the prompt, not chunk ids. Out-of-range
    positions are dropped, and an answer that cited nothing usable falls back to
    the top hit, so a grounded answer is never shown without a source.
    """
    chosen = [hits[position - 1].chunk for position in cited if 1 <= position <= len(hits)]
    if not chosen:
        chosen = [hits[0].chunk]

    return [
        Citation(
            chunk_id=chunk.id,
            excerpt=chunk.text[:EXCERPT_CHARS].strip(),
            page=chunk.page,
        )
        for chunk in chosen
    ]
