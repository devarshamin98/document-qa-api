"""Application errors and the handlers that render them as envelopes.

Every failure path ends here, so no raw stack trace or unstructured 500 body can
reach a client.
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import APIError, APIStatusError

from app.core.logging import get_request_id
from app.ingestion.loaders import (
    DocumentError,
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidDocumentError,
)

log = structlog.get_logger(__name__)


class AppError(Exception):
    """Base class for errors with a stable code and status."""

    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MissingFileError(AppError):
    code = "MISSING_FILE"
    status_code = 422


class UnsupportedFileTypeError(AppError):
    code = "UNSUPPORTED_FILE_TYPE"
    status_code = 422


class FileTooLargeError(AppError):
    code = "FILE_TOO_LARGE"
    status_code = 413


class InvalidQuestionsError(AppError):
    code = "INVALID_QUESTIONS"
    status_code = 422


class TooManyQuestionsError(AppError):
    code = "TOO_MANY_QUESTIONS"
    status_code = 422


class QuestionTooLongError(AppError):
    code = "QUESTION_TOO_LONG"
    status_code = 422


class RequestTimeoutError(AppError):
    code = "REQUEST_TIMEOUT"
    status_code = 504


class ConfigurationError(AppError):
    code = "NOT_CONFIGURED"
    status_code = 503


# Document problems are raised by `ingestion/`, which knows nothing about HTTP.
_DOCUMENT_ERRORS: dict[type[DocumentError], tuple[str, int]] = {
    InvalidDocumentError: ("INVALID_DOCUMENT", 422),
    EmptyDocumentError: ("EMPTY_DOCUMENT", 422),
    DocumentTooLargeError: ("DOCUMENT_TOO_LARGE", 413),
}


def _envelope(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": get_request_id(),
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so every error response uses the envelope."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        log.info("request_rejected", code=exc.code, status=exc.status_code)
        return _envelope(exc.code, exc.message, exc.status_code)

    @app.exception_handler(DocumentError)
    async def _document_error(_: Request, exc: DocumentError) -> JSONResponse:
        code, status_code = _DOCUMENT_ERRORS.get(type(exc), ("INVALID_DOCUMENT", 422))
        log.info("document_rejected", code=code, status=status_code)
        return _envelope(code, str(exc), status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        missing = [
            ".".join(str(part) for part in error["loc"][1:])
            for error in exc.errors()
            if error["type"] == "missing"
        ]
        if missing:
            return _envelope(
                MissingFileError.code,
                f"Missing required field(s): {', '.join(missing)}.",
                MissingFileError.status_code,
            )
        return _envelope("VALIDATION_ERROR", "The request could not be validated.", 422)

    @app.exception_handler(APIError)
    async def _openai_error(_: Request, exc: APIError) -> JSONResponse:
        status_code = getattr(exc, "status_code", None) if isinstance(exc, APIStatusError) else None
        log.error("upstream_error", upstream_status=status_code, error=type(exc).__name__)
        return _envelope(
            "UPSTREAM_ERROR",
            "The language model provider could not be reached. Please retry.",
            502,
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", error=type(exc).__name__)
        return _envelope(
            "INTERNAL_ERROR",
            "An unexpected error occurred. Quote the request_id when reporting it.",
            500,
        )
