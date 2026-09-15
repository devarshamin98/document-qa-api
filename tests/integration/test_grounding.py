"""Grounding: answers come from the document, and absent evidence says so.

What these tests can and cannot prove: the language model is faked, so they
assert nothing about answer *wording*. What they do assert is everything around
it — that retrieval surfaces the passage holding the evidence, that citations
point at passages actually retrieved rather than being invented, and that a
question with no support produces a not-found rather than a fabricated answer.
Answer quality itself needs a real model, and is checked by the live run.
"""

import json
import re

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_qa_service_provider
from app.config import Settings, get_settings
from app.core.ports import LLMResult
from app.main import create_app
from app.qa.prompts import NOT_FOUND
from app.qa.service import QAService
from app.retrieval.store import FaissStore
from tests.conftest import FIXTURES, FakeEmbeddings, FakeLLM

KB = (FIXTURES / "company_kb.json").read_bytes()

# Question -> a phrase from the knowledge base entry that answers it. If
# retrieval works, that phrase is in the passages handed to the model.
ANSWERABLE = {
    "Where are your data centres located?": "US Central",
    "Do you monitor and restrict the installation of unauthorized software?": "whitelisting",
    (
        "Is business continuity and operational resilience documentation "
        "available to authorized stakeholders?"
    ): "operational resilience documentation is available",
    "Is there a dedicated sanctions compliance officer?": "dedicated sanctions compliance officer",
    (
        "Are private keys provisioned for a unique purpose managed, and is cryptography secret?"
    ): "unique purpose",
    (
        "Does the Incident Response Plan include event reporting mechanism "
        "to support the reporting action?"
    ): "event reporting mechanism",
}

# Real questions from the sample questionnaire on topics this knowledge base
# genuinely does not cover.
UNANSWERABLE = [
    "Do transport containers protect against physical damage?",
    "Does the Hypervisor system lock accounts after 3-5 invalid login attempts?",
    "Does the Wireless Security Policy prohibit wired and wireless network "
    "connections at the same time?",
    "Is there a Quality Assurance organization that ensures software integrity "
    "for FDA-regulated systems?",
]

_PASSAGE_RE = re.compile(
    r"\[Passage ([A-Z]+)\]\n(.*?)(?=\n\n\[Passage [A-Z]+\]\n|\n\nQuestion:)", re.DOTALL
)


class PassageEchoLLM:
    """A model that can only restate the passages it was handed.

    Deliberately incapable of using outside knowledge, so a correct answer is
    proof that retrieval delivered the right passage. If retrieval regresses,
    the echoed answer stops containing the expected evidence and the test fails.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.passages_seen: list[list[tuple[str, str]]] = []

    async def complete(self, system: str, user: str) -> LLMResult:
        self.calls += 1
        passages = [(label, text.strip()) for label, text in _PASSAGE_RE.findall(user)]
        self.passages_seen.append(passages)
        if not passages:
            return LLMResult(
                text=json.dumps({"answer": NOT_FOUND, "sources": []}),
                prompt_tokens=50,
                completion_tokens=4,
            )
        body = " ".join(text for _, text in passages[:2])
        return LLMResult(
            text=json.dumps({"answer": body, "sources": [label for label, _ in passages[:2]]}),
            prompt_tokens=400,
            completion_tokens=60,
        )


@pytest.fixture
def settings() -> Settings:
    """Smaller chunks so retrieval has to discriminate between KB entries."""
    return Settings(_env_file=None, openai_api_key=None, chunk_size=350, chunk_overlap=50)


@pytest.fixture
def client_for(settings: Settings, fake_embeddings: FakeEmbeddings):
    def build(llm: object) -> AsyncClient:
        service = QAService(
            llm=llm, embeddings=fake_embeddings, store_factory=FaissStore, settings=settings
        )
        app = create_app()
        app.dependency_overrides[get_qa_service_provider] = lambda: lambda: service
        app.dependency_overrides[get_settings] = lambda: settings
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    return build


async def post(client: AsyncClient, questions: list[str], document: bytes = KB):
    return await client.post(
        "/api/v1/qa",
        files={
            "questions": ("questions.json", json.dumps(questions).encode(), "application/json"),
            "document": ("company_kb.json", document, "application/json"),
        },
    )


async def test_a_batch_of_answerable_questions_is_answered_from_the_evidence(
    client_for,
) -> None:
    llm = PassageEchoLLM()
    async with client_for(llm) as client:
        response = await post(client, list(ANSWERABLE))

    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert len(results) == len(ANSWERABLE)

    for result in results:
        evidence = ANSWERABLE[result["question"]]
        assert result["found"] is True, f"no answer for: {result['question']}"
        assert result["error"] is None
        assert evidence.lower() in result["answer"].lower(), (
            f"retrieval missed the evidence for {result['question']!r}; "
            f"expected {evidence!r} in the passages"
        )
        assert result["citations"], "a grounded answer must cite its source"

    assert llm.calls == len(ANSWERABLE), "one completion per question, none wasted"


async def test_citations_point_at_passages_that_were_actually_retrieved(
    client_for,
) -> None:
    """Guards against citations being invented rather than mapped from retrieval."""
    llm = PassageEchoLLM()
    async with client_for(llm) as client:
        response = await post(client, list(ANSWERABLE))

    # Passages are numbered by position in the prompt, so a citation is verified
    # by its text matching a passage that was actually sent, not by id equality.
    sent = [text for passages in llm.passages_seen for _, text in passages]

    checked = 0
    for result in response.json()["results"]:
        for citation in result["citations"]:
            checked += 1
            excerpt = citation["excerpt"][:60]
            assert any(passage.startswith(excerpt) for passage in sent), (
                f"cited an excerpt that was never sent to the model: {excerpt!r}"
            )
            assert citation["page"] == 1, "a JSON document is one synthetic page"

    # Without this the loop above passes vacuously whenever retrieval returns
    # nothing, which is exactly the regression it exists to catch.
    assert checked >= len(ANSWERABLE), "expected at least one citation per question"


async def test_a_batch_with_no_supporting_evidence_returns_not_found_for_every_question(
    client_for,
) -> None:
    """The contract when nothing in the document answers anything asked."""
    llm = FakeLLM()  # no canned answers, so it returns the sentinel
    async with client_for(llm) as client:
        response = await post(client, UNANSWERABLE)

    assert response.status_code == 200, "an unanswerable batch is a result, not a failure"
    results = response.json()["results"]
    assert len(results) == len(UNANSWERABLE)

    for result in results:
        assert result["found"] is False
        assert result["answer"] == NOT_FOUND
        assert result["citations"] == [], "nothing was supported, so nothing may be cited"
        assert result["error"] is None, "absent evidence is a correct answer, not a failure"

    # Token usage is still reported, so the caller can see what the attempt cost.
    assert response.json()["meta"]["tokens"]["embedding"] > 0


async def test_a_question_with_no_lexical_support_never_reaches_the_model(
    client_for,
) -> None:
    """Retrieval below the score floor short-circuits before spending a completion."""
    llm = PassageEchoLLM()
    async with client_for(llm) as client:
        question = "Do transport containers protect against physical damage?"
        response = await post(client, [question])

    result = response.json()["results"][0]
    assert result["found"] is False
    assert result["answer"] == NOT_FOUND
    assert result["error"] is None
    assert llm.calls == 0, "nothing cleared the floor, so no completion should be bought"
