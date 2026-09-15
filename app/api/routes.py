"""HTTP surface: the QA endpoint and a health check."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import QAServiceProviderDep, SettingsDep
from app.api.errors import (
    FileTooLargeError,
    InvalidQuestionsError,
    MissingFileError,
    QuestionTooLongError,
    RequestTimeoutError,
    TooManyQuestionsError,
    UnsupportedFileTypeError,
)
from app.api.schemas import (
    CitationModel,
    MetaModel,
    QAResponse,
    QuestionAnswerModel,
    TokenUsageModel,
)
from app.config import Settings
from app.ingestion.loaders import Page, load_json, load_pdf

log = structlog.get_logger(__name__)

router = APIRouter()

READ_CHUNK_BYTES = 1024 * 1024
SUPPORTED_SUFFIXES = {".pdf", ".json"}


@router.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.post("/api/v1/qa", response_model=QAResponse, tags=["qa"])
async def answer_questions(
    provider: QAServiceProviderDep,
    settings: SettingsDep,
    questions: Annotated[
        UploadFile | None, File(description="JSON file listing the questions")
    ] = None,
    document: Annotated[UploadFile | None, File(description="PDF or JSON document")] = None,
) -> QAResponse:
    """Answer every question in `questions` against `document`."""
    if questions is None or document is None:
        missing = [
            name
            for name, value in (("questions", questions), ("document", document))
            if value is None
        ]
        raise MissingFileError(f"Missing required file(s): {', '.join(missing)}.")

    suffix = Path(document.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(
            f"Unsupported document type '{suffix or 'unknown'}'. Upload a .pdf or .json file."
        )

    question_bytes = await _read_capped(questions, settings, label="questions")
    document_bytes = await _read_capped(document, settings, label="document")

    parsed_questions = _parse_questions(question_bytes, settings)

    # pypdf is blocking CPU work; keep it off the event loop.
    pages: list[Page] = await run_in_threadpool(
        _load_document, document_bytes, suffix, settings.max_pdf_pages
    )

    structlog.contextvars.bind_contextvars(
        document_type=suffix.lstrip("."), questions=len(parsed_questions)
    )

    # Constructed only once the request is known to be well formed.
    service = provider()

    try:
        result = await asyncio.wait_for(
            service.answer(pages=pages, questions=parsed_questions),
            timeout=settings.request_timeout_s,
        )
    except TimeoutError as exc:
        raise RequestTimeoutError(
            f"Request exceeded {settings.request_timeout_s:g}s. "
            "Try fewer questions or a smaller document."
        ) from exc

    return QAResponse(
        results=[
            QuestionAnswerModel(
                question=answer.question,
                answer=answer.answer,
                found=answer.found,
                error=answer.error,
                citations=[
                    CitationModel(chunk_id=c.chunk_id, excerpt=c.excerpt, page=c.page)
                    for c in answer.citations
                ],
            )
            for answer in result.answers
        ],
        meta=MetaModel(
            document_name=document.filename or "document",
            document_type=suffix.lstrip("."),
            chunks=result.chunk_count,
            latency_ms=result.latency_ms,
            tokens=TokenUsageModel(
                prompt=result.prompt_tokens,
                completion=result.completion_tokens,
                embedding=result.embedding_tokens,
            ),
        ),
    )


async def _read_capped(upload: UploadFile, settings: Settings, *, label: str) -> bytes:
    """Read an upload, aborting as soon as it exceeds the cap.

    Checked while streaming rather than after, so an oversized upload is rejected
    without ever being held in memory in full.
    """
    limit = settings.max_upload_bytes
    parts: list[bytes] = []
    total = 0
    while chunk := await upload.read(READ_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise FileTooLargeError(
                f"The {label} file exceeds the {settings.max_upload_mb:g} MB limit."
            )
        parts.append(chunk)
    return b"".join(parts)


def _parse_questions(data: bytes, settings: Settings) -> list[str]:
    """Accept either `["q1", "q2"]` or `{"questions": ["q1", ...]}`."""
    try:
        parsed = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise InvalidQuestionsError("The questions file is not valid UTF-8 text.") from exc
    except json.JSONDecodeError as exc:
        raise InvalidQuestionsError(
            f"The questions file is not valid JSON: {exc.msg} at line {exc.lineno}."
        ) from exc

    if isinstance(parsed, dict):
        raw = parsed.get("questions")
    elif isinstance(parsed, list):
        raw = parsed
    else:
        raw = None

    if not isinstance(raw, list):
        raise InvalidQuestionsError(
            'Expected a JSON list of questions, or an object with a "questions" list.'
        )
    if not all(isinstance(item, str) for item in raw):
        raise InvalidQuestionsError("Every question must be a string.")

    cleaned = [item.strip() for item in raw if item.strip()]
    if not cleaned:
        raise InvalidQuestionsError("The questions file contains no questions.")
    if len(cleaned) > settings.max_questions:
        raise TooManyQuestionsError(
            f"Max {settings.max_questions} questions per request; received {len(cleaned)}."
        )

    for index, question in enumerate(cleaned):
        if len(question) > settings.max_question_chars:
            raise QuestionTooLongError(
                f"Question {index + 1} is {len(question)} characters; "
                f"the limit is {settings.max_question_chars}."
            )
    return cleaned


def _load_document(data: bytes, suffix: str, max_pdf_pages: int) -> list[Page]:
    if suffix == ".pdf":
        return load_pdf(data, max_pages=max_pdf_pages)
    return load_json(data)
