"""Turn uploaded bytes into pages of text.

These are synchronous CPU-bound functions — callers in `api/` run them via
`run_in_threadpool` so pypdf never blocks the event loop. They raise
`DocumentError` subclasses, which `api/errors.py` maps to HTTP envelopes; this
module knows nothing about HTTP.
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PyPdfError

# Keys whose value is an opaque handle rather than content.
_ID_KEY_RE = re.compile(r"(^|[._])(id|uuid|guid|hash|checksum|etag)$", re.IGNORECASE)
# Values that carry no meaning to a reader: long unbroken hex or uuid tokens.
_OPAQUE_VALUE_RE = re.compile(
    r"^(?:[0-9a-f]{24,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)


def is_opaque_identifier(key_path: str, value: str) -> bool:
    """True for a value that is a machine handle, not readable content.

    Both halves must agree: the key reads like an identifier *and* the value is
    an unbroken hex or uuid token. Requiring both keeps a field merely named
    "id" whose value is a real sentence, and keeps a hex-looking value under a
    meaningful key. Such values are pure cost — they are embedded, they occupy
    chunk space, and no question is answered by them.
    """
    return bool(_ID_KEY_RE.search(key_path) and _OPAQUE_VALUE_RE.match(value.strip()))


@dataclass(frozen=True, slots=True)
class Page:
    """One page of extracted text. JSON documents produce a single synthetic page."""

    number: int
    text: str


class DocumentError(Exception):
    """Base class for documents we cannot use."""


class InvalidDocumentError(DocumentError):
    """Bytes could not be parsed as the declared type."""


class EmptyDocumentError(DocumentError):
    """Parsed fine but yielded no usable text (e.g. a scanned PDF)."""


class DocumentTooLargeError(DocumentError):
    """Exceeds a configured structural limit, such as page count."""


def flatten_json(obj: Any, prefix: str = "") -> Iterator[str]:
    """Flatten nested JSON into readable `key.path: value` lines.

    Dict keys join with `.`, list items get `[i]` suffixes. Empty containers and
    nulls are emitted explicitly rather than dropped, so a question about a field
    that exists but is unset can still be answered from the text.
    """
    if isinstance(obj, dict):
        if not obj:
            yield f"{prefix}: (empty object)" if prefix else "(empty object)"
            return
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json(value, child)
    elif isinstance(obj, list):
        if not obj:
            yield f"{prefix}: (empty list)" if prefix else "(empty list)"
            return
        for index, value in enumerate(obj):
            yield from flatten_json(value, f"{prefix}[{index}]")
    else:
        if obj is None:
            rendered = "null"
        elif isinstance(obj, bool):
            rendered = "true" if obj else "false"
        else:
            rendered = str(obj)
        if prefix and is_opaque_identifier(prefix, rendered):
            return
        yield f"{prefix}: {rendered}" if prefix else rendered


def iter_records(parsed: Any) -> Iterator[str]:
    """Yield one flattened block per logical record in the document.

    A record is the unit a reader would treat as self-contained: an item of a
    top-level array, or an item of an array hanging off a top-level key. Blocks
    are separated by a blank line downstream, which is the splitter's preferred
    boundary, so a record stays whole in one chunk instead of being cut by a
    character count.

    This matters for question/answer banks, where one entry's answer and the
    next entry's question would otherwise share a chunk while the entry's own
    question and answer were split across two.
    """
    if isinstance(parsed, list):
        for index, item in enumerate(parsed):
            yield "\n".join(flatten_json(item, f"[{index}]"))
        return

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if isinstance(value, list) and value:
                for index, item in enumerate(value):
                    yield "\n".join(flatten_json(item, f"{key}[{index}]"))
            else:
                yield "\n".join(flatten_json(value, str(key)))
        return

    yield "\n".join(flatten_json(parsed))


def load_json(data: bytes) -> list[Page]:
    """Parse and flatten a JSON document into one page of `key.path: value` lines."""
    try:
        parsed = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise InvalidDocumentError("Document is not valid UTF-8 text.") from exc
    except json.JSONDecodeError as exc:
        raise InvalidDocumentError(
            f"Document is not valid JSON: {exc.msg} at line {exc.lineno}."
        ) from exc

    blocks = [block.strip() for block in iter_records(parsed) if block.strip()]
    text = "\n\n".join(blocks).strip()
    if not text:
        raise EmptyDocumentError("The JSON document contains no readable content.")
    return [Page(number=1, text=text)]


def load_pdf(data: bytes, max_pages: int) -> list[Page]:
    """Extract text per page, keeping page numbers for citations.

    Pages with no extractable text are skipped; a PDF with none at all is
    rejected rather than silently answered from an empty index.
    """
    try:
        reader = PdfReader(BytesIO(data))
        page_count = len(reader.pages)
    except (PyPdfError, ValueError, OSError) as exc:
        raise InvalidDocumentError("Document could not be read as a PDF.") from exc

    if page_count > max_pages:
        raise DocumentTooLargeError(f"PDF has {page_count} pages; the limit is {max_pages}.")

    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except (PyPdfError, ValueError):
            text = ""
        if text.strip():
            pages.append(Page(number=number, text=text))

    if not pages:
        raise EmptyDocumentError(
            "No extractable text found. The PDF appears to be scanned images; OCR is not supported."
        )
    return pages
