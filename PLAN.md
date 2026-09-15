# Build Plan — 90 minute version

Five prompts, one session, no `/clear` between them. `CLAUDE.md` is the spec;
this file is the schedule and the list of deliberate deviations from it.

Every prompt ends with:

> Run `uv run ruff check . && uv run pytest -q`, fix anything failing, then
> commit with a conventional message.

---

## Deliberate cuts (these override CLAUDE.md)

Time-driven, not accidental. Name them in the README's tradeoffs section.

| Cut | Why it's safe |
|-----|---------------|
| No `GET /metrics`, no `core/metrics.py` | Rubric row 6 reads "structured logging (JSON logs) **and/or** metrics". JSON logs alone satisfy it. Latency + token usage still go in the logs and in `meta`; already struck from CLAUDE.md. |
| Content-type validated by extension only, no magic bytes | Still returns a clean 422 envelope on a wrong file type. Defence-in-depth only. |
| ~8 tests, not the exhaustive list in CLAUDE.md's testing rules | Covers the rubric's actual ask: unit tests for core logic + edge cases, plus integration tests against a mocked LLM. |
| No separate hardening pass | Obvious fixes fold into the buffer. |

Everything else in `CLAUDE.md` stands — especially the hard constraints, the
limits table, the grounding rules, and the git hygiene rules.

---

## Schedule

| Time | Phase |
|------|-------|
| 0:00–0:30 | **A** — scaffold through working endpoint |
| 0:30–0:35 | **Smoke** — one real-key run |
| 0:35–0:50 | **B** — tests |
| 0:50–0:58 | **C** — logging + Dockerfile (build backgrounded) |
| 0:58–1:08 | **D** — frontend; check the Docker build when done |
| 1:08–1:20 | **E** — README + SOC2 sample run |
| 1:20–1:30 | **Buffer** — fresh-clone check, push, email |

---

## A — 0:00–0:30 — Scaffold through working endpoint

Use plan mode for this one only. Review the plan for two minutes, then approve.

> Read CLAUDE.md and PLAN.md. Build the whole service in one pass, applying the
> deliberate cuts in PLAN.md — skip `core/metrics.py`, the `/metrics` endpoint,
> and magic-byte validation.
>
> Scaffold with uv. `.gitignore` goes in the first commit, before `git add .`,
> and must include `*.docx` — the challenge documents in the repo root contain a
> live OpenAI key in plaintext. Then: `config.py` with every limit from the
> Limits section; `.env.example` with a placeholder value only; ingestion (pypdf
> per-page loader, JSON flattener to `"key.path: value"` lines,
> `RecursiveCharacterTextSplitter` preserving page metadata); Protocols in
> `core/ports.py`; OpenAI embeddings adapter with batching (<=100 per call) and a
> FAISS store adapter; `prompts.py` with a grounded prompt that numbers context
> chunks and requires the exact sentinel `Data Not Available`; `qa/service.py`
> that ingests once, embeds all questions in one call, retrieves top-k, answers
> concurrently under a semaphore sized from config, and maps cited chunk numbers
> to citations; `POST /api/v1/qa` per the API contract with every limit enforced
> through the error envelope and a catch-all handler so nothing returns a raw
> 500; `GET /health`.
>
> `qa/service.py` must return token usage in its result object so it lands in
> `meta.tokens` — phase C only has to log it.
>
> Create `tests/conftest.py` with `FakeLLM` and `FakeEmbeddings` wired through
> FastAPI dependency overrides, plus one smoke test that the endpoint returns 200
> for a tiny JSON document.
>
> Also create the fixtures the smoke run needs: `tests/fixtures/sample.pdf` (one
> page, a few sentences of fake SOC2-style text) and
> `tests/fixtures/questions.json` with two questions, one answerable from that
> text and one not. Commit the generated `.pdf` as a static fixture so the test
> suite has no PDF-authoring dependency at runtime — either hand-write a minimal
> PDF (a text stream in a single content object is plain ASCII) or generate it
> with reportlab kept as a dev-only dependency.
>
> Before committing, list each requirement from this prompt and confirm it is
> implemented. Name anything you skipped.

**Verify:** server starts, `/health` returns ok, smoke test passes with no
`OPENAI_API_KEY` set.

---

## Smoke — 0:30–0:35 — One real-key run

Do this yourself, not through a prompt. The fakes hide every real-integration
bug: wrong model string, embedding dimension mismatch, response parsing, async
client misuse. Finding those at 1:05 leaves no room to fix them.

```bash
cp .env.example .env          # paste the real key into .env
uv run uvicorn app.main:app &
curl -s -F questions=@tests/fixtures/questions.json \
        -F document=@tests/fixtures/sample.pdf \
        localhost:8000/api/v1/qa | jq .
```

Two questions against a one-page PDF. Costs well under a cent. Confirm
`meta.tokens` is populated and at least one answer has a citation.

If it fails, fix it now — everything downstream assumes this path works.

---

## B — 0:35–0:50 — Tests

> Read CLAUDE.md testing rules. Add exactly these tests using the fakes.
>
> Unit: JSON flattener with nested objects and arrays; chunker overlap and a
> document smaller than one chunk; question-file parsing for both accepted shapes
> and one malformed input; the `MAX_QUESTIONS` limit returning the right error
> code.
>
> Integration: happy path PDF (extend `tests/fixtures/sample.pdf` to two pages,
> or add a second fixture beside it); a not-found question
> returning `found=false` and the sentinel; an oversized upload returning the 413
> envelope; a wrong document extension returning the 422 envelope.
>
> For the oversized test, monkeypatch `MAX_UPLOAD_MB` to a tiny value and upload
> a few KB. Do not commit a 20 MB fixture.

**Verify:** `uv run pytest -q` passes with no `OPENAI_API_KEY` set and no network.

---

## C — 0:50–0:58 — Logging + Docker

> Read CLAUDE.md observability and container sections. Configure structlog for
> JSON to stdout. Add request-id middleware logging method, path, status, and
> `latency_ms`. In the QA service, log per-request document type, chunk count,
> question count, token usage, and total latency; per-question latency and found
> flag. Never log the key, document text, or more than the first 80 chars of a
> question.
>
> Write a multi-stage Dockerfile with a non-root user and uv, and a
> `docker-compose.yml` that reads `.env` and exposes 8000.

Then start the build in the background and move straight to D:

```bash
docker compose build > /tmp/dockerbuild.log 2>&1 &
```

**Verify (at the end of D):** `docker compose up` serves `/health`; every stdout
line parses as JSON.

---

## D — 0:58–1:08 — Frontend

> Read CLAUDE.md. Create `app/static/index.html` served at `GET /`. Plain HTML
> and vanilla JS, no build step, no frameworks, under 120 lines: two file inputs
> (questions JSON, document PDF/JSON), a submit button, loading text, and a
> results table with question / answer / found badge / citations. Display error
> envelope messages inline.

**Verify:** upload the sample files in a browser and see results. Then check
`/tmp/dockerbuild.log`.

---

## E — 1:08–1:20 — README + sample run

First, run the sample SOC2 PDF with the five sample questions once via the UI or
curl, and save the response JSON. Check `meta.tokens` and stop if cumulative
spend approaches $3.

> Write `README.md` for a reviewer with five minutes: a one-paragraph overview,
> an ASCII architecture sketch, quickstart with both `docker compose` and `uv`,
> the API contract with a curl example and this real sample response (paste it),
> the limits table, how grounding and citations work, how to run tests and why
> they need no API key, logging notes, and five bullets of design tradeoffs —
> FAISS in-memory vs a hosted vector DB, per-request ingestion vs caching, the
> concurrency bound, gpt-4o-mini, and what you would add with more time (name the
> cut items from this file). No marketing tone.

---

## Buffer — 1:20–1:30

- [ ] `git log -p | grep -cE "sk-proj-[A-Za-z0-9_-]{20,}"` prints `0`
      (matches a real key, not the bare prefix — this file mentions the prefix itself)
- [ ] `git ls-files | grep -c docx` prints `0`
- [ ] Fresh clone, add `.env` only, `docker compose up --build`, hit the UI once
- [ ] Push, email the repo link to shruti@zania.ai

---

## Tripwires

- **`uv sync` builds faiss-cpu from source** (rather than pulling a wheel): do
  not debug it. Swap in a numpy cosine store — roughly 15 lines behind the
  existing `VectorStore` protocol, and the README tradeoffs section already has a
  line for it.
- **Phase A overruns past 0:40:** cut D entirely. The frontend is 5 points; a
  broken endpoint is 60.
- **Docker build fails:** ship the Dockerfile anyway and say so in the README. A
  correct Dockerfile that a grader does not run still reads as done; burning 20
  minutes on it does not.

## Budget

$5 lifetime. Only the 0:30 smoke run and the phase E sample run touch the real
key — a few cents total. Tests never hit OpenAI.
