# Document QA API

A FastAPI service that answers a list of questions about an uploaded document
(PDF or JSON) using retrieval-augmented generation with `gpt-4o-mini`. Upload a
questions file and a document; get back one structured answer per question, each
carrying the passages it came from, or an explicit `Data-Not-Found` when the
document does not support an answer. Unanswerable questions are treated as a
correct outcome rather than a failure — a wrong answer is worse than no answer.

## Architecture

```
          upload (multipart)
                 │
     ┌───────────▼────────────┐
     │  api/   routes, schemas│  validation, limits, error envelope
     │         errors, deps   │  the only layer that knows about HTTP
     └───────────┬────────────┘
                 │ pages
     ┌───────────▼────────────┐
     │  ingestion/            │  pypdf per page │ JSON → "key.path: value"
     │  loaders, chunking     │  split per page, page numbers preserved
     └───────────┬────────────┘
                 │ chunks
     ┌───────────▼────────────┐        ┌──────────────────────┐
     │  qa/service.py         │───────▶│ retrieval/embeddings │──▶ OpenAI
     │  ingest once,          │        │ batched ≤100/call    │
     │  embed once,           │        └──────────────────────┘
     │  answer concurrently   │        ┌──────────────────────┐
     │  under a semaphore     │───────▶│ retrieval/store      │  faiss, cosine
     │                        │        └──────────────────────┘
     │                        │        ┌──────────────────────┐
     │                        │───────▶│ llm/client + prompts │──▶ OpenAI
     └───────────┬────────────┘        └──────────────────────┘
                 │ answers + citations + token usage
                 ▼
            JSON response
```

Everything that talks to a third party implements a Protocol in
`app/core/ports.py` (`LLMClient`, `EmbeddingsClient`, `VectorStore`) and is
injected. That is what lets the entire test suite run offline against fakes, and
what makes swapping faiss for a hosted vector database a single-file change.

## Quickstart

### Docker

```bash
cp .env.example .env        # add your OPENAI_API_KEY
docker compose up --build   # UI at http://localhost:8000
```

`.env` is optional — without a key the service starts and answers `503
NOT_CONFIGURED`, so a reviewer can boot it and look around before providing one.

### Local

```bash
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload
```

Open <http://localhost:8000> for the upload UI, or
<http://localhost:8000/docs> for the generated OpenAPI docs.

## API

### `POST /api/v1/qa`

`multipart/form-data` with two files:

| Field | Accepts |
|-------|---------|
| `questions` | `.json` — either `["q1", "q2"]` or `{"questions": ["q1", ...]}` |
| `document` | `.pdf` or `.json` |

```bash
curl -s -F questions=@tests/fixtures/questions.json \
        -F document=@tests/fixtures/sample.pdf \
        http://localhost:8000/api/v1/qa | jq .
```

A real response from that command, against the two-page fixture:

```json
{
    "results": [
        {
            "question": "Which cloud providers does Acme Corp rely on?",
            "answer": "Acme Corp relies on Amazon Web Services (AWS) and Google Cloud Platform (GCP) as its infrastructure providers.",
            "found": true,
            "error": null,
            "citations": [
                {
                    "chunk_id": 1,
                    "excerpt": "Acme Corp - System and Organization Controls (SOC 2 Type II)\nSection 1: Infrastructure and Subservice Organizations\nAcme Corp relies on Amazon Web Services (AWS) and Google Cloud\nPlatform (GCP) as its infrastructure providers. No other clou",
                    "page": 1
                }
            ]
        },
        {
            "question": "What is the CEO's shoe size?",
            "answer": "Data-Not-Found",
            "found": false,
            "error": null,
            "citations": []
        }
    ],
    "meta": {
        "document_name": "sample.pdf",
        "document_type": "pdf",
        "chunks": 2,
        "latency_ms": 1358,
        "tokens": {
            "prompt": 904,
            "completion": 50,
            "embedding": 215
        }
    }
}
```

Each result has three distinguishable states, because a failure and a correct
"the document does not say" are opposite facts:

| `found` | `error` | Meaning |
|---------|---------|---------|
| `true` | `null` | Answered from the document, with citations. |
| `false` | `null` | The document genuinely does not cover this. **A correct answer.** |
| `false` | `"TIMEOUT"` | That question failed. Others in the batch are unaffected. |

A per-question timeout degrades only that question; the request still returns
`200` with every other answer intact, rather than discarding work already paid
for in tokens.

### `GET /health`

```json
{"status": "ok"}
```

### Errors

Every 4xx and 5xx response uses one envelope, and no unhandled exception can
escape as a raw stack trace:

```json
{"error": {"code": "TOO_MANY_QUESTIONS",
           "message": "Max 50 questions per request; received 73.",
           "request_id": "1a79d16ab205"}}
```

The `request_id` also appears on the `x-request-id` response header and on every
log line for that request, so a user-reported failure can be traced to its logs.

| Code | Status |
|------|--------|
| `MISSING_FILE`, `UNSUPPORTED_FILE_TYPE`, `INVALID_QUESTIONS` | 422 |
| `TOO_MANY_QUESTIONS`, `QUESTION_TOO_LONG` | 422 |
| `INVALID_DOCUMENT`, `EMPTY_DOCUMENT` | 422 |
| `FILE_TOO_LARGE`, `DOCUMENT_TOO_LARGE` | 413 |
| `REQUEST_TIMEOUT` | 504 |
| `UPSTREAM_ERROR` | 502 |
| `NOT_CONFIGURED` | 503 |
| `INTERNAL_ERROR` | 500 |

## Limits

All live in `app/config.py` and are overridable by environment variable
(`MAX_QUESTIONS=10 uv run uvicorn app.main:app`):

| Setting | Default | Purpose |
|---------|---------|---------|
| `MAX_UPLOAD_MB` | 20 | Checked while streaming the upload, not after |
| `MAX_PDF_PAGES` | 300 | Rejected before any text extraction |
| `MAX_QUESTIONS` | 50 | Bounds the number of completions per request |
| `MAX_QUESTION_CHARS` | 1000 | Bounds prompt size |
| `MAX_CONCURRENT_LLM_CALLS` | 8 | Semaphore over the answer fan-out |
| `LLM_TIMEOUT_S` | 30 | Per question |
| `REQUEST_TIMEOUT_S` | 180 | Whole request |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Characters |
| `TOP_K` | 5 | Passages retrieved per question |
| `MIN_SCORE` | 0.15 | Cosine floor below which retrieval counts as a miss |

## Grounding and citations

1. The document is chunked **per page**, so a chunk never straddles a page
   boundary and every chunk carries a real page number.
2. Chunks are embedded once per request and indexed in faiss
   (`IndexFlatIP` over L2-normalized vectors, which is cosine similarity).
3. All questions are embedded in **one** call, then each retrieves its own
   top-`TOP_K` passages.
4. If the best passage scores below `MIN_SCORE`, no completion is requested at
   all — there is nothing to ground an answer in, so spending a call to be told
   so would be waste.
5. Otherwise the passages go into the prompt under lettered headings
   (`[Passage A]`, `[Passage B]`, ...), which instructs the model to use only
   those passages, to answer with exactly `Data-Not-Found` when they are
   insufficient, and to return JSON: `{"answer": "...", "sources": ["A", "C"]}`.
6. Those letters map back to chunk ids on our side, producing citations with
   page number and excerpt. `found` is false when the sentinel comes back.

Both details in step 5 were forced by measurement rather than chosen up front:

- **Letters, not numbers.** Passages were first labelled by chunk id, then by
  position. Flattened JSON is full of bracketed indices like
  `knowledge_base[14]`, and the model kept citing those instead: 7 of 17 answers
  cited a passage outside the set of five it was given. Letters cannot collide
  with an array index, and out-of-range citations went to 0.
- **JSON, not a `SOURCES:` line.** Free text was unreliable — the model
  variously omitted the line or invented a number. Requesting a JSON object and
  enforcing it with `response_format` removed that class of failure entirely.

An out-of-range or missing citation still falls back to the top-scoring passage,
so a grounded answer is never shown without a source.

Temperature is pinned to `0`: the task is to restate what the passages say, and
sampling variance there is a defect rather than creativity.

## Tests

```bash
uv run pytest -q                              # 60 tests
uv run ruff check . && uv run ruff format --check .
```

**No API key and no network are needed.** `FakeLLM` and `FakeEmbeddings`
implement the ports and are injected via FastAPI dependency overrides, and an
autouse fixture deletes `OPENAI_API_KEY` from the environment so a key that
happens to be present cannot be used by accident.

`FakeEmbeddings` hashes words into buckets rather than returning random vectors,
so lexical overlap produces genuine cosine similarity — otherwise every score
would sit near zero and retrieval assertions would pass for the wrong reason.

Unit tests cover JSON flattening, PDF loading (including the scanned-PDF and
page-limit paths), chunk overlap and page-boundary integrity, question parsing
and every limit. Integration tests cover both happy paths, the sentinel path,
each error envelope, and a timeout.

`tests/integration/test_grounding.py` runs a batch of real security-questionnaire
questions against a knowledge-base fixture. The fake model there can only restate
the passages it is handed, so a correct answer proves retrieval delivered the
right passage; a second test asserts every citation points at a passage that was
actually retrieved rather than invented. A matching batch of questions the
fixture does not cover asserts the opposite: `200`, every result `found: false`
with `error: null` and no citations, and questions that fall below the score
floor never reach the model at all. Both were verified by mutation — inverting
retrieval and disabling sentinel detection each fail the tests that should
catch them. Two tests assert properties the endpoint
cannot show on its own: that LLM calls stay within the semaphore bound while
genuinely overlapping, and that the document is embedded exactly once no matter
how many questions arrive.

## Evaluation

The sample answer set shipped with the challenge doubles as ground truth, so
answer quality is measured rather than asserted. `evaluate.py` posts all 19
questions and compares each verdict (Yes / No / N/A / Data-Not-Found) against
the expected answer.

| Evidence | Verdict agreement | Answers with a citation | Cost |
|----------|------------------|------------------------|------|
| Sample document (contains the answers) | 15/19 | 17/17 | $0.004 |
| A knowledge base covering ~half the topics | 13/19 | 12/12 | $0.004 |

Both runs agree on the row that matters most: where the answer key says
`Data-Not-Found`, so do we.

Read the two numbers together rather than as a score out of 19:

- On the first run, **all four misses are `N/A` rows** — "this does not apply to
  us", a verdict distinct from "not in the document" that this API deliberately
  does not model. Excluding them, agreement is 15/15.
- On the second run, the extra misses are questions whose evidence simply is not
  in that smaller knowledge base, and the service correctly returned
  `Data-Not-Found` for them. That looks like a lower score while being the
  desired behaviour.

The second run is the stronger result, for a reason the score hides: the model
demonstrably knew those answers, because it produced them from the fuller
document in the first run. Given a knowledge base that omits them, it declined
instead of recalling them. That is the property retrieval-grounded answering
exists to provide.

## Observability

Every line on stdout is a single JSON object, uvicorn's own output included:

```json
{"method": "POST", "path": "/api/v1/qa", "status": 200, "latency_ms": 4210,
 "event": "request_finished", "request_id": "cfb859eb0476", "level": "info",
 "timestamp": "2026-09-15T19:03:53.183647Z"}
```

Per request the service logs document type, chunk count, question count, token
usage and latency. Per question it logs latency, `found`, and the top retrieval
score. Document text is never logged and questions are truncated to 80
characters, both asserted by tests.

Token usage is also returned in `meta.tokens` on every response, so cost is
visible to the caller without reading logs.

## Design decisions and tradeoffs

- **faiss in-memory, rebuilt per request.** At document scale an exact flat
  index searches in microseconds, so there is no recall tradeoff and nothing to
  operate. The cost is that an identical document is re-embedded on every
  request. A content-hash cache in front of ingestion, or a persistent store
  keyed by document, is the first thing I would add — it needs a `/documents`
  endpoint and an eviction policy to be worth doing properly.
- **Per-request ingestion rather than a document lifecycle.** The brief asks for
  two files in one request, so the service is stateless. That keeps deployment
  trivial and makes every request reproducible, at the cost of the re-embedding
  above.
- **A concurrency bound of 8.** Questions are answered with `asyncio.gather`
  under a semaphore. Unbounded fan-out on a 50-question request would hit rate
  limits and turn one slow batch into a thundering herd; 8 keeps latency close
  to serial-divided-by-8 without that risk.
- **`gpt-4o-mini` at temperature 0, capped at 500 completion tokens.** Required
  by the brief, and the right shape for the task: extraction and restatement,
  not generation. The cap bounds the cost of a runaway answer.
- **LangChain is used for text splitting only.** `RecursiveCharacterTextSplitter`
  earns its place; the OpenAI SDK and faiss are driven directly. Going through
  `langchain-openai` would have hidden `response.usage`, which is exactly what
  `meta.tokens` needs, and an LCEL chain would have fought the per-question
  semaphore, per-question timeout and citation parsing.

### What I would add with more time

- A content-hash cache so re-uploading the same document skips ingestion.
- Hybrid retrieval — BM25 fused with dense similarity via reciprocal rank
  fusion. Embeddings blur exact tokens like `APM`, `SOC 2` or `us-east-1`, which
  are precisely what questionnaire questions ask about. It costs no LLM calls.
- Section-aware chunking: extract the heading hierarchy from the PDF and prefix
  each chunk with its breadcrumb before embedding, so citations read
  "Section IV, CC7.2, p.42" rather than a page number alone.
- Question decomposition for multi-part questions — "do you have criteria, and
  what are your SLAs?" is two retrievals — gated behind a cheap detector so it
  does not spend a call on every question.
- Deliberately cut for time: a `/metrics` endpoint (latency and token usage go
  to the JSON logs instead), magic-byte content sniffing beyond the extension
  check, and an exhaustive test matrix beyond the ~48 tests here.
