"""Score the API against an answer key, to measure answer quality rather than assert it.

    uv run uvicorn app.main:app --port 8000          # in one shell
    uv run python scripts/evaluate.py \
        --csv "Sample JSON.xlsx - Sheet1.csv" \
        --document path/to/evidence.json

The answer key is a CSV with `question` and `answer` columns — the sample answer
set supplied with the challenge has exactly this shape. It is not included in
this repository: it is the challenge author's material, not part of the
deliverable, so point `--csv` at your own copy.

Each answer is reduced to a verdict (YES / NO / N/A / NOT_FOUND / PROSE) and
compared with the expected one. Verdict agreement is a coarse measure — it says
an answer points the same direction as the key, not that it is well worded. The
most informative line is not-found agreement: whether the service says "no
evidence" exactly where the key does.
"""

import argparse
import csv
import json
import pathlib
import re
import sys
import urllib.request

NOT_FOUND = "Data-Not-Found"


def verdict(answer: str, found: bool | None = None) -> str:
    """Reduce an answer to a comparable verdict."""
    if found is False:
        return "NOT_FOUND"
    text = answer.strip()
    if text == NOT_FOUND or text.lower().startswith("data-not-found"):
        return "NOT_FOUND"
    if re.match(r"^n/?a\b", text, re.IGNORECASE):
        return "N/A"
    if re.match(r"^yes\b", text, re.IGNORECASE):
        return "YES"
    # \b matches before "." and ",", so "No." and "No, we..." both count.
    if re.match(r"^no\b", text, re.IGNORECASE):
        return "NO"
    return "PROSE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="Sample JSON.xlsx - Sheet1.csv")
    parser.add_argument("--document", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    questions = [r["question"].strip() for r in rows]
    expected = [verdict(r["answer"]) for r in rows]

    document = pathlib.Path(args.document)
    boundary = "----evalboundary"
    parts = []
    for field, filename, payload, ctype in (
        ("questions", "questions.json", json.dumps(questions).encode(), "application/json"),
        ("document", document.name, document.read_bytes(), "application/json"),
    ):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
            f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
            + payload
            + b"\r\n"
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        f"{args.url}/api/v1/qa",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)

    results = payload["results"]
    print(f"{'#':>3}  {'EXPECTED':<10} {'OURS':<10} {'':<3} CITES  QUESTION")
    agree = notfound_right = notfound_expected = cited = answered = 0
    for index, (want, result) in enumerate(zip(expected, results, strict=True)):
        got = verdict(result["answer"], result["found"])
        match = want == got
        agree += match
        if want == "NOT_FOUND":
            notfound_expected += 1
            notfound_right += got == "NOT_FOUND"
        if result["found"]:
            answered += 1
            cited += bool(result["citations"])
        print(
            f"{index:>3}  {want:<10} {got:<10} {'ok' if match else 'XX':<3} "
            f"{len(result['citations']):>5}  {result['question'][:52]}"
        )

    total = len(results)
    print(f"\nverdict agreement      {agree}/{total}  ({agree / total:.0%})")
    if notfound_expected:
        print(f"not-found agreement    {notfound_right}/{notfound_expected}")
    print(f"answered with a citation {cited}/{answered}" if answered else "nothing answered")
    meta = payload["meta"]
    tokens = meta["tokens"]
    cost = tokens["prompt"] / 1e6 * 0.15 + tokens["completion"] / 1e6 * 0.60
    print(f"\n{meta['chunks']} chunks, {meta['latency_ms']} ms")
    print(f"tokens: {tokens}  ->  approx ${cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
