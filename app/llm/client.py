"""OpenAI chat adapter implementing the `LLMClient` port."""

from openai import AsyncOpenAI

from app.core.ports import LLMResult


class OpenAILLM:
    """Single-turn completions against a chat model.

    Temperature is pinned to 0: the job is to restate what the passages say, and
    sampling variance there is a bug, not creativity. `max_tokens` caps the cost
    of a runaway answer.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(self, system: str, user: str) -> LLMResult:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            # The prompt asks for a JSON object; enforcing it here means a
            # malformed citation list cannot reach the parser.
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = response.usage
        return LLMResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
