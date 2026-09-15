"""Log output is machine-readable and carries the fields an operator needs."""

import json
import logging
import sys
from io import StringIO

import pytest
import structlog
from httpx import AsyncClient

from app.config import Settings
from app.core.logging import configure_logging
from app.ingestion.loaders import Page
from app.qa.service import QUESTION_LOG_CHARS, QAService
from app.retrieval.store import FaissStore
from tests.conftest import FakeEmbeddings, FakeLLM


@pytest.fixture
def captured_stdout(monkeypatch: pytest.MonkeyPatch):
    """Point the logging handler at a buffer, then restore global logging state."""
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    configure_logging()
    yield stream
    monkeypatch.undo()
    logging.getLogger().handlers.clear()
    configure_logging()


def test_every_log_line_is_a_single_json_object(captured_stdout: StringIO) -> None:
    structlog.get_logger("test").info("something_happened", answer=42)

    lines = [line for line in captured_stdout.getvalue().splitlines() if line.strip()]
    assert lines, "expected at least one log line"

    for line in lines:
        payload = json.loads(line)  # raises if the line is not valid JSON
        assert "timestamp" in payload
        assert "level" in payload

    assert json.loads(lines[-1])["event"] == "something_happened"
    assert json.loads(lines[-1])["answer"] == 42


async def test_access_log_records_method_path_status_and_latency(client: AsyncClient) -> None:
    with structlog.testing.capture_logs() as logs:
        await client.get("/health")

    finished = [entry for entry in logs if entry["event"] == "request_finished"]
    assert len(finished) == 1, "exactly one access line per request"

    entry = finished[0]
    assert entry["method"] == "GET"
    assert entry["path"] == "/health"
    assert entry["status"] == 200
    assert isinstance(entry["latency_ms"], int)


async def test_request_logs_carry_token_usage_and_document_type(
    client: AsyncClient, sample_pdf_bytes: bytes
) -> None:
    with structlog.testing.capture_logs() as logs:
        await client.post(
            "/api/v1/qa",
            files={
                "questions": (
                    "q.json",
                    b'["Which cloud providers does Acme rely on?"]',
                    "application/json",
                ),
                "document": ("sample.pdf", sample_pdf_bytes, "application/pdf"),
            },
        )

    completed = next(entry for entry in logs if entry["event"] == "qa_completed")
    assert completed["tokens_prompt"] > 0
    assert completed["tokens_embedding"] > 0
    assert completed["chunks"] >= 2
    assert completed["questions"] == 1
    assert isinstance(completed["latency_ms"], int)


async def test_logs_truncate_questions_and_never_carry_document_text(
    settings: Settings,
) -> None:
    """Questions may contain customer specifics; documents certainly do."""
    secret = "CONFIDENTIAL-CANARY-STRING"
    long_question = "alpha beta gamma " + "x" * 300
    service = QAService(
        llm=FakeLLM(),
        embeddings=FakeEmbeddings(),
        store_factory=FaissStore,
        settings=settings,
    )

    with structlog.testing.capture_logs() as logs:
        await service.answer(
            pages=[Page(number=1, text=f"alpha beta gamma {secret} " * 20)],
            questions=[long_question],
        )

    rendered = json.dumps(logs)
    assert secret not in rendered, "document text must never reach the logs"

    answered = next(entry for entry in logs if entry["event"] == "question_answered")
    assert len(answered["question"]) <= QUESTION_LOG_CHARS
    assert answered["question"] == long_question[:QUESTION_LOG_CHARS]
