"""Request and response models for the public API."""

from pydantic import BaseModel, Field


class CitationModel(BaseModel):
    chunk_id: int = Field(description="1-based id of the passage the answer came from")
    excerpt: str = Field(description="Leading text of that passage")
    page: int = Field(description="Page the passage came from; 1 for JSON documents")


class QuestionAnswerModel(BaseModel):
    """One question's result.

    `found` and `error` together distinguish three outcomes: answered
    (`found=true`), genuinely not covered by the document (`found=false`,
    `error=null` — a correct answer), and failed (`error` set).
    """

    question: str
    answer: str
    found: bool
    error: str | None = Field(
        default=None,
        description='Short failure code such as "TIMEOUT"; null when the question was processed',
    )
    citations: list[CitationModel] = Field(default_factory=list)


class TokenUsageModel(BaseModel):
    prompt: int
    completion: int
    embedding: int


class MetaModel(BaseModel):
    document_name: str
    document_type: str
    chunks: int
    latency_ms: int
    tokens: TokenUsageModel


class QAResponse(BaseModel):
    results: list[QuestionAnswerModel]
    meta: MetaModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    """Shape of every 4xx and 5xx body."""

    error: ErrorDetail
