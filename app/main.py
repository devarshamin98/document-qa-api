"""Application factory."""

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import router
from app.core.logging import RequestIdMiddleware, configure_logging


def create_app() -> FastAPI:
    """Build the app: logging, middleware, error handlers, routes."""
    configure_logging()

    app = FastAPI(
        title="Document QA API",
        description="Answers questions about an uploaded PDF or JSON document.",
        version="0.1.0",
    )
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
