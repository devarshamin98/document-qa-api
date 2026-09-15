"""The grounded-answer prompt and its response parser.

This is the 10-point grounding row of the rubric. Two things have to be
unambiguous: that the model may use nothing but the supplied passages, and that
an unsupported question gets the exact sentinel back rather than a hedge.
"""

import re
from collections.abc import Sequence

from app.core.ports import Chunk

NOT_FOUND = "Data-Not-Found"

_SOURCES_RE = re.compile(r"^\s*SOURCES\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = f"""You answer questions about one document, using ONLY the numbered \
passages supplied with each question.

Rules:
1. Use only the passages. Never use outside knowledge and never guess. If the
   passages disagree with what you believe to be true, follow the passages.
2. If the passages do not contain enough information to answer, reply with
   exactly this and nothing else:
   {NOT_FOUND}
   Do not explain, do not hedge, do not say the document "does not appear to"
   cover it, and do not offer a partial answer. The exact string is required.
3. Otherwise answer in one to four sentences of plain prose. Be specific: name
   the providers, regions, timeframes or controls the passages state.
4. End every reply with a final line naming the passage numbers you used:
   SOURCES: 1, 3
   If you replied "{NOT_FOUND}", end with an empty list:
   SOURCES:
"""


def build_user_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    """Render the numbered passages followed by the question."""
    passages = "\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in chunks)
    return f"Passages:\n\n{passages}\n\nQuestion: {question}"


def parse_answer(raw: str) -> tuple[str, list[int]]:
    """Split a reply into prose and cited passage numbers.

    Tolerant by design: a missing or malformed `SOURCES:` line costs citations,
    not the answer, because the prose is the part the user reads.
    """
    text = raw.strip()
    matches = list(_SOURCES_RE.finditer(text))
    if not matches:
        return text, []

    last = matches[-1]
    answer = text[: last.start()].strip()
    cited = [int(number) for number in re.findall(r"\d+", last.group(1))]

    # De-duplicate while preserving the order the model gave.
    seen: set[int] = set()
    ordered = [n for n in cited if not (n in seen or seen.add(n))]
    return answer or text, ordered


def is_not_found(answer: str) -> bool:
    """True when the model returned the sentinel, ignoring case and punctuation."""
    return answer.strip().rstrip(".").casefold() == NOT_FOUND.casefold()
