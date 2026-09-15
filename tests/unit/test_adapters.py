"""The OpenAI adapters, against a stub client.

These cover the code that only ever runs against the real API: request shape,
batch splitting, ordering, and usage accounting. The batching path in particular
cannot be reached by any document small enough to use as a fixture — the limit
is 100 texts per call — so without this it ships unexercised.
"""

from types import SimpleNamespace

import pytest

from app.llm.client import OpenAILLM
from app.retrieval.embeddings import OpenAIEmbeddings


class StubEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def create(self, *, model: str, input: list[str]):
        self.calls.append(list(input))
        # One deterministic vector per input, so ordering is checkable.
        data = [SimpleNamespace(embedding=[float(len(text)), 0.0, 1.0]) for text in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=len(input)))


class StubChatAPI:
    def __init__(self, content: str = '{"answer": "Yes.", "sources": ["A"]}') -> None:
        self.content = content
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=321, completion_tokens=45),
        )


def embeddings_client(api: StubEmbeddingsAPI, batch_size: int = 100) -> OpenAIEmbeddings:
    client = SimpleNamespace(embeddings=api)
    return OpenAIEmbeddings(client, model="text-embedding-3-small", batch_size=batch_size)


async def test_texts_are_split_into_batches_within_the_api_limit() -> None:
    api = StubEmbeddingsAPI()
    texts = [f"chunk number {index}" for index in range(250)]

    result = await embeddings_client(api).embed_texts(texts)

    assert [len(batch) for batch in api.calls] == [100, 100, 50]
    assert len(result.vectors) == 250
    assert result.tokens == 250, "usage is summed across every batch"


async def test_vectors_come_back_in_the_order_the_texts_went_in() -> None:
    """Batches are issued concurrently, so ordering is a real risk."""
    api = StubEmbeddingsAPI()
    texts = ["a" * length for length in range(1, 121)]  # length identifies the text

    result = await embeddings_client(api).embed_texts(texts)

    assert [vector[0] for vector in result.vectors] == [float(n) for n in range(1, 121)]


async def test_an_empty_input_makes_no_api_call() -> None:
    api = StubEmbeddingsAPI()

    result = await embeddings_client(api).embed_texts([])

    assert api.calls == []
    assert result.vectors == [] and result.tokens == 0


@pytest.mark.parametrize("batch_size", [1, 7, 100])
async def test_every_text_is_embedded_exactly_once(batch_size: int) -> None:
    api = StubEmbeddingsAPI()
    texts = [f"text {index}" for index in range(30)]

    await embeddings_client(api, batch_size=batch_size).embed_texts(texts)

    submitted = [text for batch in api.calls for text in batch]
    assert submitted == texts


async def test_llm_sends_the_documented_request_and_reports_usage() -> None:
    api = StubChatAPI()
    client = SimpleNamespace(chat=SimpleNamespace(completions=api))

    result = await OpenAILLM(client, model="gpt-4o-mini").complete("SYSTEM", "USER")

    assert api.kwargs["model"] == "gpt-4o-mini"
    assert api.kwargs["temperature"] == 0.0, "grounded restatement must not sample"
    assert api.kwargs["response_format"] == {"type": "json_object"}
    assert [message["role"] for message in api.kwargs["messages"]] == ["system", "user"]
    assert result.text == '{"answer": "Yes.", "sources": ["A"]}'
    assert (result.prompt_tokens, result.completion_tokens) == (321, 45)


async def test_llm_tolerates_a_response_with_no_content_or_usage() -> None:
    api = StubChatAPI(content=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=api))

    result = await OpenAILLM(client, model="gpt-4o-mini").complete("S", "U")

    assert result.text == ""
    assert result.prompt_tokens == 321
