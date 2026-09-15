"""Shared fixtures and fakes.

The whole suite runs offline: `FakeLLM` and `FakeEmbeddings` implement the ports
in `app/core/ports.py` and are injected through FastAPI dependency overrides, so
nothing here can reach OpenAI even if a key is present in the environment.
"""

import asyncio
import hashlib
import math
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_qa_service_provider
from app.config import Settings, get_settings
from app.core.ports import EmbedResult, LLMResult
from app.main import create_app
from app.qa.prompts import NOT_FOUND
from app.qa.service import QAService
from app.retrieval.store import FaissStore

FIXTURES = Path(__file__).parent / "fixtures"

_WORD_RE = re.compile(r"[a-z0-9]+")


@pytest.fixture(autouse=True)
def _never_use_a_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt and braces: no test may pick up a real key from the environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class FakeEmbeddings:
    """Deterministic bag-of-words vectors.

    Words hash into buckets, so lexical overlap produces genuine cosine
    similarity. That matters: random vectors would sit near zero and trip the
    `min_score` floor, making retrieval tests pass or fail for the wrong reason.
    """

    DIM = 64

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    async def embed_texts(self, texts: Sequence[str]) -> EmbedResult:
        items = list(texts)
        self.calls += 1
        self.batch_sizes.append(len(items))
        return EmbedResult(
            vectors=[self.vector(text) for text in items],
            tokens=sum(len(text.split()) for text in items),
        )

    @classmethod
    def vector(cls, text: str) -> list[float]:
        buckets = [0.0] * cls.DIM
        for word in _WORD_RE.findall(text.lower()):
            digest = hashlib.sha256(word.encode()).hexdigest()[:8]
            buckets[int(digest, 16) % cls.DIM] += 1.0
        norm = math.sqrt(sum(value * value for value in buckets))
        return [value / norm for value in buckets] if norm else buckets


class FakeLLM:
    """Canned answers keyed by substring of the prompt.

    Answers are keyed against the question alone, not the whole prompt: the
    prompt also carries the retrieved passages, so matching on all of it would
    let document text trigger a canned answer for an unrelated question.

    Tracks concurrent entries so a test can assert the service honours its
    semaphore bound by inspecting `max_concurrent` — deterministic, no sleeps
    required to observe it.
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        *,
        delay: float = 0.0,
        sources: str = "1",
    ) -> None:
        self.answers = dict(answers or {})
        self.delay = delay
        self.sources = sources
        self.calls = 0
        self.prompts: list[str] = []
        self.max_concurrent = 0
        self._active = 0

    async def complete(self, system: str, user: str) -> LLMResult:
        self.calls += 1
        self.prompts.append(user)
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            question = user.rsplit("Question:", 1)[-1].strip().lower()
            for key, answer in self.answers.items():
                if key.lower() in question:
                    return LLMResult(
                        text=f"{answer}\nSOURCES: {self.sources}",
                        prompt_tokens=120,
                        completion_tokens=24,
                    )
            return LLMResult(text=f"{NOT_FOUND}\nSOURCES:", prompt_tokens=120, completion_tokens=4)
        finally:
            self._active -= 1


@pytest.fixture
def settings() -> Settings:
    """Defaults, isolated from any local `.env`."""
    return Settings(_env_file=None, openai_api_key=None)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(
        answers={
            "cloud providers": "Acme Corp relies on AWS and Google Cloud Platform.",
            "notifying": "Affected customers are notified within 24 hours of confirmation.",
        }
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def qa_service(fake_llm: FakeLLM, fake_embeddings: FakeEmbeddings, settings: Settings) -> QAService:
    return QAService(
        llm=fake_llm,
        embeddings=fake_embeddings,
        store_factory=FaissStore,
        settings=settings,
    )


@pytest.fixture
def app(qa_service: QAService, settings: Settings):
    application = create_app()
    application.dependency_overrides[get_qa_service_provider] = lambda: lambda: qa_service
    application.dependency_overrides[get_settings] = lambda: settings
    return application


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as instance:
        yield instance


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return (FIXTURES / "sample.pdf").read_bytes()
