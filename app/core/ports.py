"""Domain types and the Protocols that keep OpenAI and faiss at arm's length.

Everything that talks to a third party implements one of these, so tests swap in
fakes without patching. Token counts ride along in the result objects, which is
what lets the QA service report `meta.tokens` without a metrics module.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage. `id` is 1-based so prompt numbering maps onto citations."""

    id: int
    text: str
    page: int


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: int
    excerpt: str
    page: int


@dataclass(frozen=True, slots=True)
class EmbedResult:
    vectors: list[list[float]]
    tokens: int


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class EmbeddingsClient(Protocol):
    """Turns text into vectors, reporting tokens consumed."""

    async def embed_texts(self, texts: Sequence[str]) -> EmbedResult: ...


class LLMClient(Protocol):
    """Single-turn completion."""

    async def complete(self, system: str, user: str) -> LLMResult: ...


class VectorStore(Protocol):
    """Similarity search over chunks. One instance per request."""

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    async def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]: ...
