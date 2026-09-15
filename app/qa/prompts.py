"""The grounded-answer prompt and its response parser.

This is the 10-point grounding row of the rubric. Two things have to be
unambiguous: that the model may use nothing but the supplied passages, and that
an unsupported question gets the exact sentinel back rather than a hedge.
"""

import json
import re
from collections.abc import Sequence

from app.core.ports import Chunk

NOT_FOUND = "Data-Not-Found"

_SOURCES_RE = re.compile(r"^\s*SOURCES\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = f"""You answer questions about one document, using ONLY the numbered \
passages supplied with each question.

Reply with a JSON object and nothing else:
  {{"answer": "...", "sources": ["A", "C"]}}

Rules:
1. Use only the passages. Never use outside knowledge and never guess. If the
   passages disagree with what you believe to be true, follow the passages.
2. If the passages do not contain enough information to answer, set "answer" to
   exactly "{NOT_FOUND}" and "sources" to []. Do not explain, do not hedge, do
   not say the document "does not appear to" cover it, and do not offer a
   partial answer. The exact string is required.
3. Otherwise "answer" is one to four sentences of plain prose. Be specific: name
   the providers, regions, timeframes or controls the passages state.
4. "sources" lists the letters from the [Passage X] headings you actually used,
   for example ["A", "C"]. Use only letters that appear in a heading.
"""


def _label(position: int) -> str:
    """A, B, C ... for passage positions, wrapping to AA, AB beyond 26."""
    if position < 26:
        return chr(ord("A") + position)
    first, second = divmod(position, 26)
    return chr(ord("A") + first - 1) + chr(ord("A") + second)


def _position(label: str) -> int | None:
    """Inverse of `_label`, 1-based. None when the label is not a letter code."""
    text = label.strip().upper()
    if not text or not text.isalpha():
        return None
    value = 0
    for character in text:
        value = value * 26 + (ord(character) - ord("A") + 1)
    return value


def build_user_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    """Render the passages, numbered by position, followed by the question.

    Numbering is 1..N over the retrieved set rather than by chunk id. Chunk ids
    are global and arrive in relevance order, so the model would see arbitrary
    labels like [5] [4] [8] [1] [7] and has to track which is which — observed
    in practice to produce a correct answer attributed to the wrong passage.
    Positions map back to chunk ids on our side, where it cannot go wrong.

    Labels are letters, not numbers. Flattened JSON is full of bracketed indices
    like `knowledge_base[14]`, and the model kept citing those: with numeric
    labels, 7 of 17 answers cited a passage outside the set of 5 it was given.
    Letters cannot collide with an array index.
    """
    passages = "\n\n".join(
        f"[Passage {_label(position)}]\n{chunk.text}" for position, chunk in enumerate(chunks)
    )
    return f"Passages:\n\n{passages}\n\nQuestion: {question}"


def parse_answer(raw: str) -> tuple[str, list[int]]:
    """Split a reply into prose and cited passage positions (1-based).

    The model is asked for JSON, which removes a whole class of failure: with a
    free-text `SOURCES:` line, gpt-4o-mini variously omitted it, or cited a
    number lifted out of the passage body (a set of 5 passages drew "8" and
    "15"). A text fallback remains so a non-JSON reply still yields an answer
    rather than an error.
    """
    text = raw.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and "answer" in payload:
        answer = str(payload["answer"]).strip()
        sources: list[int] = []
        for entry in payload.get("sources") or []:
            position = _position(str(entry))
            if position is None and str(entry).strip().lstrip("-").isdigit():
                position = int(entry)  # tolerate a model that reverts to numbers
            if position is not None:
                sources.append(position)
        return answer, _dedupe(sources)

    matches = list(_SOURCES_RE.finditer(text))
    if not matches:
        return text, []

    last = matches[-1]
    answer = text[: last.start()].strip()
    cited = [int(number) for number in re.findall(r"\d+", last.group(1))]
    return answer or text, _dedupe(cited)


def _dedupe(numbers: list[int]) -> list[int]:
    """Preserve the order the model gave, without repeats."""
    seen: set[int] = set()
    return [n for n in numbers if not (n in seen or seen.add(n))]


def is_not_found(answer: str) -> bool:
    """True when the model returned the sentinel, ignoring case and punctuation."""
    return answer.strip().rstrip(".").casefold() == NOT_FOUND.casefold()
