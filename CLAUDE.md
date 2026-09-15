# Document QA API — Take-Home Challenge

## What this is

A production-quality FastAPI service that answers a list of questions against a
document (PDF or JSON) using retrieval-augmented generation with `gpt-4o-mini`.
It is a graded take-home. Every decision should be traceable to the rubric below.

## Rubric (100 pts) — optimize for this, not for cleverness

| Pts | Category | What the grader is looking for |
|-----|----------|--------------------------------|
| 15 | Backend correctness | Endpoints match spec, valid JSON out, PDF **and** JSON docs, many questions per request |
| 15 | Error handling | Clear 4xx validation errors, limits (file size, page count, question count), timeouts, friendly messages, no stack traces leaking |
| 15 | Code quality | Separation of concerns, small modules, typed, readable |
| 15 | Tests | Unit tests for core logic + edge cases; **integration test of the endpoint with a mocked LLM** |
| 15 | Performance | Async I/O, concurrent question answering, efficient chunking/retrieval, no wasted LLM calls |
| 10 | Container + observability | Dockerfile, docker-compose, structured JSON logs, latency + token usage (logged, not a metrics endpoint) |
| 10 | Grounding | Answers cite retrieved chunks; returns an explicit "not found" when unsupported |
| 5  | Frontend | Minimal page: upload two files, show results. No polish needed |

~60% of points are engineering hygiene. Treat tests, validation, logging, and
Docker as first-class deliverables, not cleanup tasks.

## Hard constraints

- Model: `gpt-4o-mini` only. Never gpt-4, gpt-4-turbo, or 16k-token variants.
- Embeddings: `text-embedding-3-small`.
- The API key has a **$5 lifetime budget**. Tests must NEVER hit OpenAI. All
  LLM/embedding calls go through an interface that is mocked in tests.
- `OPENAI_API_KEY` is read from environment / `.env` only. Never hardcode it,
  never log it, never write it into any tracked file. `.env` is gitignored.
- Do not add features beyond the rubric. No auth, no database, no multi-tenant,
  no streaming, no chat history.

## Stack

- Python 3.12, `uv` for dependency management (`pyproject.toml` + `uv.lock`)
- FastAPI + uvicorn, Pydantic v2
- `langchain-text-splitters` for chunking (`RecursiveCharacterTextSplitter`). That is
  the only LangChain dependency — there is no LCEL chain. Retrieval and prompting
  are ours, behind the Protocols in `core/ports.py`.
- `faiss-cpu` driven directly (`IndexFlatIP` over normalized vectors = cosine),
  in-memory behind the `VectorStore` port. No external service to run — reviewers
  can `docker compose up` and go.
- `pypdf` for PDF text extraction
- `openai` Python SDK directly (`AsyncOpenAI`). Chosen over `langchain-openai` so
  `response.usage` is available per call and feeds `meta.tokens` without a callback
  handler.
- `structlog` for JSON logs
- `pytest`, `pytest-asyncio`, `httpx` for tests
- `ruff` for lint + format

## Architecture

```
app/
  main.py               # FastAPI app factory, middleware, router mounting
  config.py             # Settings via pydantic-settings (limits, model names, timeouts)
  api/
    routes.py           # POST /api/v1/qa, GET /health
    schemas.py          # Request/response Pydantic models
    errors.py           # Exception handlers -> consistent JSON error envelope
    deps.py             # FastAPI dependencies (settings, QA service)
  core/
    ports.py            # Protocols: LLMClient, EmbeddingsClient, VectorStore
    logging.py          # structlog config, request-id middleware
  ingestion/
    loaders.py          # PDF -> text, JSON -> flattened text with key paths
    chunking.py         # Recursive character splitter, configurable size/overlap
  retrieval/
    store.py            # faiss IndexFlatIP adapter implementing VectorStore port
    embeddings.py       # AsyncOpenAI embeddings adapter, batched (<=100/call)
  llm/
    client.py           # AsyncOpenAI chat adapter implementing LLMClient port
  qa/
    prompts.py          # Grounded-answer prompt, "not found" sentinel
    service.py          # Orchestration: ingest once -> retrieve per question -> answer concurrently
  static/
    index.html          # Minimal upload UI
tests/
  unit/                 # loaders, chunking, JSON flattening, schemas, limits
  integration/          # endpoint test with fake LLM + fake embeddings
  fixtures/             # small sample PDF, sample JSON doc, sample questions
```

Rules:
- `api/` knows about HTTP. `qa/`, `retrieval/`, `ingestion/` do not.
- Anything that talks to OpenAI implements a Protocol in `core/ports.py` and is
  injected, so tests swap in fakes with zero patching gymnastics.
- One function, one job. If a module passes ~150 lines, split it.
- Full type hints. Docstrings on public functions only, one line unless the
  behavior is non-obvious.

## API contract

`POST /api/v1/qa` — `multipart/form-data`
- `questions`: JSON file. Accepts either `["q1", "q2"]` or `{"questions": ["q1", ...]}`.
- `document`: `.pdf` or `.json`.

Response `200`:
```json
{
  "results": [
    {
      "question": "Which cloud providers do you rely on?",
      "answer": "AWS and Google Cloud.",
      "found": true,
      "error": null,
      "citations": [{"chunk_id": 12, "excerpt": "…", "page": 4}]
    },
    {
      "question": "What is your CEO's shoe size?",
      "answer": "Data-Not-Found",
      "found": false,
      "error": null,
      "citations": []
    }
  ],
  "meta": {
    "document_name": "soc2.pdf",
    "document_type": "pdf",
    "chunks": 87,
    "latency_ms": 4210,
    "tokens": {"prompt": 18234, "completion": 612, "embedding": 40211}
  }
}
```

A result has three distinguishable states, because a failure and a correct
"not in the document" are opposite facts:

| `found` | `error` | Meaning |
|---------|---------|---------|
| `true`  | `null`  | Answered from the document, with citations. |
| `false` | `null`  | The document genuinely does not cover this. A **correct** answer. |
| `false` | `"TIMEOUT"` | That question failed. Other questions in the batch are unaffected. |

Error envelope (all 4xx/5xx):
```json
{"error": {"code": "TOO_MANY_QUESTIONS", "message": "Max 50 questions per request; received 73.", "request_id": "1a79d16ab205"}}
```

`GET /health` → `{"status": "ok"}`

## Limits (all in `config.py`, all overridable by env var)

- `MAX_UPLOAD_MB=20`, `MAX_PDF_PAGES=300`, `MAX_QUESTIONS=50`
- `MAX_QUESTION_CHARS=1000`, `MAX_CONCURRENT_LLM_CALLS=8`
- `LLM_TIMEOUT_S=30` per question, `REQUEST_TIMEOUT_S=180` overall
- `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`, `TOP_K=5`

Exceeding a limit is a `413` or `422` with the envelope above, never a 500.

## Grounding rules

- Prompt instructs the model to answer **only** from provided context and to reply
  with the exact sentinel `Data-Not-Found` if the context doesn't support an
  answer. `found` is derived from that sentinel plus a retrieval-score floor.
- Every context chunk is numbered in the prompt; the model is asked to reference
  chunk numbers it used. Those map back to `citations`.
- Never fabricate. A "not found" is a correct answer, not a failure.

## Performance rules

- Ingest + embed the document **once** per request. Never per question.
- Embed chunks in batches (≤100 per call). Embed all questions in one call.
- Answer questions concurrently with `asyncio.gather` bounded by a semaphore.
- All I/O is async. No `time.sleep`, no sync OpenAI client, no blocking file
  reads in request handlers (use `run_in_threadpool` for pypdf).

## Observability

- Every log line is JSON via structlog: `timestamp, level, event, request_id, ...`
- Log per request: document type, chunk count, question count, total latency,
  token usage. Log per question: latency, found/not-found, top retrieval score.
- Never log document content, question text beyond first 80 chars, or the API key.

## Testing rules

- `tests/conftest.py` provides `FakeLLM` (returns canned answers keyed by
  substring, or the sentinel) and `FakeEmbeddings` (deterministic hash-based
  vectors) and wires them via FastAPI dependency overrides.
- Unit test targets: PDF loader, JSON flattener (nested objects, arrays, empty
  values), chunker (overlap, tiny docs, huge docs), question-file parsing (both
  shapes, malformed), every limit, error envelope shape.
- Integration test targets: happy path PDF, happy path JSON, not-found question,
  too many questions, oversized file, wrong file type, missing file, LLM timeout.
- `uv run pytest` must pass with **no network** and no `OPENAI_API_KEY` set.

## Commands

```bash
uv sync                          # install
uv run uvicorn app.main:app --reload
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
docker compose up --build        # serves on :8000, UI at /
```

## Git hygiene

- Commit after each phase in `PLAN.md` with a conventional message
  (`feat:`, `test:`, `chore:`).
- Run `ruff` and `pytest` before every commit.
- `.gitignore` must include `.env`, `.venv`, `__pycache__`, `*.pyc`, `.pytest_cache`,
  `*.docx`, `.DS_Store`. The challenge `.docx` files in the repo root contain a
  live OpenAI key in plaintext — they must never be committed. Write `.gitignore`
  in the very first commit, before `git add .`.
