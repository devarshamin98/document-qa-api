"""QAService behaviour that the HTTP layer cannot show: concurrency and reuse."""

from app.config import Settings
from app.ingestion.loaders import Page
from app.qa.service import QAService
from app.retrieval.store import FaissStore
from tests.conftest import FakeEmbeddings, FakeLLM

TERMS = "alpha beta gamma delta epsilon zeta"


def build_service(llm: FakeLLM, embeddings: FakeEmbeddings, settings: Settings) -> QAService:
    return QAService(llm=llm, embeddings=embeddings, store_factory=FaissStore, settings=settings)


async def test_llm_calls_are_concurrent_but_bounded_by_the_semaphore(
    settings: Settings,
) -> None:
    settings.max_concurrent_llm_calls = 3
    llm = FakeLLM(delay=0.02)
    embeddings = FakeEmbeddings()
    pages = [Page(number=1, text=f"{TERMS} " * 20)]
    questions = [f"{TERMS} question {index}?" for index in range(10)]

    await build_service(llm, embeddings, settings).answer(pages=pages, questions=questions)

    assert llm.calls == 10
    # Bounded...
    assert llm.max_concurrent <= 3
    # ...but genuinely parallel, not a serial loop that happens to stay under the cap.
    assert llm.max_concurrent > 1


async def test_the_document_is_embedded_once_regardless_of_question_count(
    settings: Settings,
) -> None:
    llm = FakeLLM()
    embeddings = FakeEmbeddings()
    pages = [Page(number=1, text=f"{TERMS} " * 20)]
    questions = [f"{TERMS} question {index}?" for index in range(8)]

    result = await build_service(llm, embeddings, settings).answer(pages=pages, questions=questions)

    # Exactly two embedding calls: the document once, then all questions in one
    # batch. Anything more means work is being repeated per question.
    assert embeddings.calls == 2
    assert embeddings.batch_sizes[1] == 8
    assert result.embedding_tokens > 0


async def test_a_question_below_the_score_floor_skips_the_llm_entirely(
    settings: Settings,
) -> None:
    settings.min_score = 0.99
    llm = FakeLLM(answers={"alpha": "Should never be reached."})
    pages = [Page(number=1, text=f"{TERMS} " * 20)]

    result = await build_service(llm, FakeEmbeddings(), settings).answer(
        pages=pages, questions=["something entirely unrelated to the document"]
    )

    assert llm.calls == 0, "retrieval missed, so there is nothing to spend a completion on"
    assert result.answers[0].found is False
    assert result.answers[0].error is None


def test_excerpt_starts_where_the_question_is_answered() -> None:
    """A chunk can hold several records; the excerpt should show the relevant one."""
    from app.qa.service import _excerpt

    chunk = (
        "[0].topic: Software installation\n"
        "[0].answer: Application whitelisting is enforced.\n"
        "[1].topic: Network protocols\n"
        "[1].answer: All data in transit uses TLS 1.3.\n"
    )

    excerpt = _excerpt(chunk, "Which protocols protect data in transit?")

    assert excerpt.startswith("[1].topic: Network protocols")


def test_excerpt_falls_back_to_the_start_when_nothing_matches() -> None:
    from app.qa.service import _excerpt

    chunk = "[0].topic: Software installation\n[0].answer: Whitelisting is enforced."

    assert _excerpt(chunk, "zzz qqq").startswith("[0].topic")
