"""Application settings.

Every limit lives here and is overridable by environment variable (matching is
case-insensitive, so `MAX_UPLOAD_MB=5` sets `max_upload_mb`).
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---------------------------------------------------------
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Upload limits --------------------------------------------------
    # float so tests can shrink it below 1 MB without a huge fixture.
    max_upload_mb: float = 20.0
    max_pdf_pages: int = 300
    max_questions: int = 50
    max_question_chars: int = 1000

    # --- Chunking and retrieval -----------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    # Cosine floor below which retrieval is treated as a miss, so a weakly
    # matched chunk cannot be dressed up as a grounded answer.
    min_score: float = 0.15
    embed_batch_size: int = 100

    # --- Concurrency and timeouts ---------------------------------------
    max_concurrent_llm_calls: int = 8
    llm_timeout_s: float = 30.0
    request_timeout_s: float = 180.0

    @property
    def max_upload_bytes(self) -> int:
        """Upload cap in bytes."""
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance; overridden in tests via dependency injection."""
    return Settings()
