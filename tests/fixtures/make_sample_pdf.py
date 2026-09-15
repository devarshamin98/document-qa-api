"""One-off generator for `sample.pdf`.

Stdlib only, by design: the committed PDF is a static fixture, so the test suite
needs no PDF-authoring dependency. Kept in the repo so the binary fixture has
visible provenance and can be regenerated.

    uv run python tests/fixtures/make_sample_pdf.py
"""

from pathlib import Path

PAGES: list[list[str]] = [
    [
        "Acme Corp - System and Organization Controls (SOC 2 Type II)",
        "",
        "Section 1: Infrastructure and Subservice Organizations",
        "",
        "Acme Corp relies on Amazon Web Services (AWS) and Google Cloud",
        "Platform (GCP) as its infrastructure providers. No other cloud",
        "providers are used to host production workloads.",
        "",
        "The primary data center region is us-east-1 (Northern Virginia).",
        "Encrypted backups are replicated to eu-west-1 (Ireland) every",
        "six hours and retained for 35 days.",
    ],
    [
        "Section 2: Incident Response and Customer Notification",
        "",
        "Acme Corp maintains formally defined criteria for notifying",
        "customers of any security incident affecting the confidentiality,",
        "integrity or availability of their data. Affected customers are",
        "notified within 24 hours of incident confirmation.",
        "",
        "Section 3: Monitoring",
        "",
        "The service is monitored using Application Performance Monitoring",
        "(APM) and End User Monitoring (EUM). Digital Experience Monitoring",
        "(DEM) is not performed as part of the monitoring process.",
    ],
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    body = "\n".join(f"({_escape(line)}) Tj T*" for line in lines)
    return f"BT\n/F1 11 Tf\n72 720 Td\n16 TL\n{body}\nET".encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Assemble a minimal but valid PDF: catalog, page tree, contents, font."""
    count = len(pages)
    font_id = 3 + 2 * count
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{2 + i} 0 R" for i in range(1, count + 1))
            + f"] /Count {count} >>"
        ).encode("latin-1"),
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, lines in enumerate(pages, start=1):
        page_id = 2 + index
        contents_id = 2 + count + index
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {contents_id} 0 R >>"
        ).encode("latin-1")
        stream = _content_stream(lines)
        objects[contents_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n".encode("latin-1") + objects[object_id] + b"\nendobj\n"

    xref_offset = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for object_id in range(1, size):
        out += f"{offsets[object_id]:010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n").encode(
        "latin-1"
    )
    return bytes(out)


if __name__ == "__main__":
    target = Path(__file__).parent / "sample.pdf"
    target.write_bytes(build_pdf(PAGES))
    print(f"wrote {target} ({target.stat().st_size} bytes, {len(PAGES)} pages)")
