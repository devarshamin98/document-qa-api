"""OpenAI embeddings adapter implementing the `EmbeddingsClient` port."""

import asyncio
from collections.abc import Sequence

from openai import AsyncOpenAI

from app.core.ports import EmbedResult


class OpenAIEmbeddings:
    """Batched embeddings over the async OpenAI client.

    The API caps inputs per call, so texts are split into batches and the batches
    are issued concurrently. `asyncio.gather` preserves order, and so does the
    API within a batch, so returned vectors line up with the input texts.
    """

    def __init__(self, client: AsyncOpenAI, *, model: str, batch_size: int = 100) -> None:
        self._client = client
        self._model = model
        self._batch_size = max(1, batch_size)

    async def embed_texts(self, texts: Sequence[str]) -> EmbedResult:
        items = list(texts)
        if not items:
            return EmbedResult(vectors=[], tokens=0)

        batches = [
            items[start : start + self._batch_size]
            for start in range(0, len(items), self._batch_size)
        ]
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))

        vectors: list[list[float]] = []
        tokens = 0
        for batch_vectors, batch_tokens in results:
            vectors.extend(batch_vectors)
            tokens += batch_tokens
        return EmbedResult(vectors=vectors, tokens=tokens)

    async def _embed_batch(self, batch: list[str]) -> tuple[list[list[float]], int]:
        response = await self._client.embeddings.create(model=self._model, input=batch)
        vectors = [item.embedding for item in response.data]
        tokens = response.usage.total_tokens if response.usage else 0
        return vectors, tokens
