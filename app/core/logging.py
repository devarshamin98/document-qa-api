"""structlog JSON logging, request-id propagation, and the access log.

Everything on stdout is one JSON object per line, including uvicorn's own
output: stdlib loggers are routed through the same formatter, so a log collector
never has to cope with two formats interleaved.
"""

import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "x-request-id"

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

log = structlog.get_logger(__name__)


def get_request_id() -> str:
    """Current request id, or `-` outside a request."""
    return _request_id.get()


def configure_logging(level: int = logging.INFO) -> None:
    """Emit one JSON object per line on stdout, for our logs and uvicorn's alike."""
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Module-level loggers would otherwise cache this processor chain at
        # import time, so tests could never swap in a capturing one. The cost is
        # negligible next to a network call to OpenAI.
        cache_logger_on_first_use=False,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own plain-text handlers; hand its records to ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # RequestIdMiddleware already emits a richer access line carrying the
    # request_id; uvicorn's would be a second entry for the same request.
    logging.getLogger("uvicorn.access").disabled = True


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind it to the log context, and emit the access log."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        token = _request_id.set(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status = 500  # what the client gets if the handler raises

        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            # Logged before the context is torn down, so the line carries the
            # request_id, and on every path including an unhandled exception.
            log.info(
                "request_finished",
                method=request.method,
                path=request.url.path,
                status=status,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            structlog.contextvars.clear_contextvars()
            _request_id.reset(token)
