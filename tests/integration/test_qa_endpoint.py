"""Endpoint behaviour against the fakes: happy paths, limits, failure modes."""

import json

import pytest
from httpx import AsyncClient

from app.config import Settings

PDF_QUESTIONS = json.dumps(
    [
        "Which cloud providers does Acme Corp rely on?",
        "What is the CEO's favourite colour?",
    ]
).encode()


def upload(document_name: str, document_bytes: bytes, questions: bytes) -> dict:
    return {
        "questions": ("questions.json", questions, "application/json"),
        "document": (document_name, document_bytes, "application/octet-stream"),
    }


async def test_happy_path_pdf(client: AsyncClient, sample_pdf_bytes: bytes) -> None:
    response = await client.post(
        "/api/v1/qa", files=upload("sample.pdf", sample_pdf_bytes, PDF_QUESTIONS)
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [result["question"] for result in body["results"]] == json.loads(PDF_QUESTIONS)

    answered = body["results"][0]
    assert answered["found"] is True
    assert answered["error"] is None
    assert "AWS" in answered["answer"]

    citation = answered["citations"][0]
    assert citation["page"] == 1, "the cloud providers passage is on page 1 of the fixture"
    assert citation["chunk_id"] >= 1
    assert citation["excerpt"]

    meta = body["meta"]
    assert meta["document_type"] == "pdf"
    assert meta["document_name"] == "sample.pdf"
    assert meta["chunks"] >= 2, "a two-page fixture yields at least one chunk per page"
    assert meta["tokens"]["prompt"] > 0
    assert meta["tokens"]["embedding"] > 0


async def test_unanswerable_question_returns_the_sentinel_not_a_guess(
    client: AsyncClient, sample_pdf_bytes: bytes
) -> None:
    response = await client.post(
        "/api/v1/qa", files=upload("sample.pdf", sample_pdf_bytes, PDF_QUESTIONS)
    )

    assert response.status_code == 200
    unanswered = response.json()["results"][1]

    assert unanswered["found"] is False
    assert unanswered["answer"] == "Data-Not-Found"
    assert unanswered["citations"] == []
    # error stays null: the document genuinely does not cover this, which is a
    # correct answer rather than a failure.
    assert unanswered["error"] is None


async def test_multiple_questions_are_all_answered(
    client: AsyncClient, sample_pdf_bytes: bytes
) -> None:
    questions = json.dumps(
        [
            "Which cloud providers does Acme Corp rely on?",
            "What are the criteria for notifying customers of an incident?",
            "What is the CEO's favourite colour?",
        ]
    ).encode()

    response = await client.post(
        "/api/v1/qa", files=upload("sample.pdf", sample_pdf_bytes, questions)
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    assert [result["found"] for result in results] == [True, True, False]


async def test_json_document_is_supported(client: AsyncClient) -> None:
    document = json.dumps(
        {"infrastructure": {"cloud providers": ["AWS", "Google Cloud Platform"]}}
    ).encode()
    questions = json.dumps(["Which cloud providers are used?"]).encode()

    response = await client.post("/api/v1/qa", files=upload("vendor.json", document, questions))

    assert response.status_code == 200
    assert response.json()["meta"]["document_type"] == "json"


async def test_oversized_upload_is_rejected_with_the_413_envelope(
    client: AsyncClient, settings: Settings
) -> None:
    # Shrink the limit rather than committing a 20 MB fixture.
    settings.max_upload_mb = 0.001  # ~1 KB
    oversized = json.dumps({"padding": "x" * 5000}).encode()

    response = await client.post(
        "/api/v1/qa", files=upload("big.json", oversized, b'["Anything?"]')
    )

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "FILE_TOO_LARGE"
    assert "document" in error["message"]
    assert error["request_id"]


async def test_unsupported_document_type_is_rejected_with_the_422_envelope(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/v1/qa", files=upload("notes.txt", b"plain text", b'["Anything?"]')
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "UNSUPPORTED_FILE_TYPE"
    assert ".txt" in error["message"]


async def test_missing_document_is_rejected_with_the_422_envelope(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/qa",
        files={"questions": ("questions.json", b'["Anything?"]', "application/json")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MISSING_FILE"


async def test_too_many_questions_is_rejected_with_the_right_code(
    client: AsyncClient, settings: Settings
) -> None:
    settings.max_questions = 2
    questions = json.dumps([f"Question {index}?" for index in range(5)]).encode()

    response = await client.post("/api/v1/qa", files=upload("vendor.json", b'{"a": 1}', questions))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "TOO_MANY_QUESTIONS"
    assert "received 5" in error["message"]


async def test_scanned_pdf_is_rejected_rather_than_answered_from_nothing(
    client: AsyncClient,
) -> None:
    from tests.fixtures.make_sample_pdf import build_pdf

    response = await client.post(
        "/api/v1/qa", files=upload("scanned.pdf", build_pdf([[]]), b'["Anything?"]')
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "EMPTY_DOCUMENT"


@pytest.mark.parametrize("payload", [b"not json at all", b'{"questions": "not a list"}'])
async def test_malformed_questions_file_is_rejected(client: AsyncClient, payload: bytes) -> None:
    response = await client.post("/api/v1/qa", files=upload("vendor.json", b'{"a": 1}', payload))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_QUESTIONS"


async def test_llm_timeout_degrades_one_question_without_failing_the_request(
    client: AsyncClient, settings: Settings, fake_llm, sample_pdf_bytes: bytes
) -> None:
    """A timeout is a failure, and must be distinguishable from 'not in the document'."""
    settings.llm_timeout_s = 0.01
    fake_llm.delay = 0.5

    response = await client.post(
        "/api/v1/qa",
        files=upload(
            "sample.pdf",
            sample_pdf_bytes,
            json.dumps(["Which cloud providers does Acme Corp rely on?"]).encode(),
        ),
    )

    assert response.status_code == 200, "one slow question must not sink the whole batch"
    result = response.json()["results"][0]
    assert result["error"] == "TIMEOUT"
    assert result["found"] is False
    assert result["answer"] == "Data-Not-Found"
