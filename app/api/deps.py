"""FastAPI dependencies — the single seam tests override."""

from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI

from app.api.errors import ConfigurationError
from app.config import Settings, get_settings
from app.llm.client import OpenAILLM
from app.qa.service import QAService
from app.retrieval.embeddings import OpenAIEmbeddings
from app.retrieval.store import FaissStore

SettingsDep = Annotated[Settings, Depends(get_settings)]


def build_qa_service(settings: Settings) -> QAService:
    """Wire the real OpenAI-backed adapters."""
    if settings.openai_api_key is None:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    return QAService(
        llm=OpenAILLM(client, model=settings.llm_model),
        embeddings=OpenAIEmbeddings(
            client,
            model=settings.embedding_model,
            batch_size=settings.embed_batch_size,
        ),
        store_factory=FaissStore,
        settings=settings,
    )


@lru_cache
def get_qa_service() -> QAService:
    """One service (and one HTTP connection pool) for the process lifetime."""
    return build_qa_service(get_settings())


def get_qa_service_provider() -> Callable[[], QAService]:
    """Hand back the factory rather than the service.

    FastAPI resolves dependencies before the handler body runs. Injecting the
    service directly would mean a missing API key surfaces as 503 on a request
    that should have been rejected as a 422 for, say, an unsupported file type.
    Deferring construction keeps validation errors ahead of configuration ones.
    """
    return get_qa_service


QAServiceProviderDep = Annotated[Callable[[], QAService], Depends(get_qa_service_provider)]
