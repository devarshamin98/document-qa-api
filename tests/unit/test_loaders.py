"""JSON flattening and document loading."""

import json

import pytest

from app.ingestion.loaders import (
    EmptyDocumentError,
    InvalidDocumentError,
    flatten_json,
    load_json,
    load_pdf,
)


def flatten(obj: object) -> list[str]:
    return list(flatten_json(obj))


def test_nested_objects_join_keys_with_dots() -> None:
    assert flatten({"vendor": {"security": {"soc2": "Type II"}}}) == [
        "vendor.security.soc2: Type II"
    ]


def test_arrays_are_indexed() -> None:
    assert flatten({"providers": ["AWS", "GCP"]}) == [
        "providers[0]: AWS",
        "providers[1]: GCP",
    ]


def test_arrays_of_objects_keep_both_paths() -> None:
    assert flatten({"regions": [{"name": "us-east-1", "primary": True}]}) == [
        "regions[0].name: us-east-1",
        "regions[0].primary: true",
    ]


def test_empty_containers_and_nulls_are_stated_not_dropped() -> None:
    # A field that exists but is unset is itself an answer, so it must survive
    # into the text the model sees.
    assert flatten({"subprocessors": [], "dpo": None, "extras": {}}) == [
        "subprocessors: (empty list)",
        "dpo: null",
        "extras: (empty object)",
    ]


def test_load_json_produces_one_page_of_flattened_lines() -> None:
    pages = load_json(json.dumps({"a": {"b": 1}, "c": [2]}).encode())
    assert len(pages) == 1
    assert pages[0].number == 1
    # Top-level entries are separated by a blank line: that is the splitter's
    # preferred boundary, so a record is not cut by a character count.
    assert pages[0].text == "a.b: 1\n\nc[0]: 2"


def test_each_item_of_a_top_level_array_is_its_own_record() -> None:
    document = [{"q": "first", "a": "yes"}, {"q": "second", "a": "no"}]

    blocks = load_json(json.dumps(document).encode())[0].text.split("\n\n")

    assert blocks == ["[0].q: first\n[0].a: yes", "[1].q: second\n[1].a: no"]


def test_items_of_an_array_under_a_key_are_records_too() -> None:
    """The common shape: a knowledge base hanging off a named key."""
    document = {"company": "Acme", "entries": [{"topic": "one"}, {"topic": "two"}]}

    blocks = load_json(json.dumps(document).encode())[0].text.split("\n\n")

    assert blocks == ["company: Acme", "entries[0].topic: one", "entries[1].topic: two"]


def test_a_record_keeps_its_fields_together_through_chunking() -> None:
    """The point of the whole exercise: one questionnaire row, one chunk."""
    from app.ingestion.chunking import chunk_pages

    document = [
        {"question": f"Question number {index}?", "answer": "Yes", "comments": "x" * 200}
        for index in range(6)
    ]
    chunks = chunk_pages(load_json(json.dumps(document).encode()), chunk_size=400, chunk_overlap=50)

    for index in range(6):
        holders = {c.id for c in chunks if f"[{index}].question:" in c.text}
        answers = {c.id for c in chunks if f"[{index}].comments:" in c.text}
        assert holders & answers, f"row {index} was split across chunks"


def test_load_json_rejects_malformed_json() -> None:
    with pytest.raises(InvalidDocumentError):
        load_json(b"{not json")


def test_load_json_rejects_a_document_with_no_content() -> None:
    with pytest.raises(EmptyDocumentError):
        load_json(b'""')


def test_load_pdf_returns_text_per_page(sample_pdf_bytes: bytes) -> None:
    pages = load_pdf(sample_pdf_bytes, max_pages=300)
    assert [page.number for page in pages] == [1, 2]
    assert "Amazon Web Services" in pages[0].text
    assert "24 hours" in pages[1].text


def test_load_pdf_rejects_a_document_over_the_page_limit(sample_pdf_bytes: bytes) -> None:
    with pytest.raises(Exception, match="2 pages"):
        load_pdf(sample_pdf_bytes, max_pages=1)


def test_load_pdf_rejects_a_pdf_with_no_text_layer() -> None:
    """A scanned PDF must fail loudly rather than yield an empty index."""
    from tests.fixtures.make_sample_pdf import build_pdf

    with pytest.raises(EmptyDocumentError, match="scanned"):
        load_pdf(build_pdf([[]]), max_pages=300)


def test_load_pdf_rejects_bytes_that_are_not_a_pdf() -> None:
    with pytest.raises(InvalidDocumentError):
        load_pdf(b"this is not a pdf", max_pages=300)


def test_opaque_identifiers_are_left_out_of_the_evidence() -> None:
    """A uuid under an id key answers no question and costs tokens to embed."""
    document = {
        "id": "53148c6413ac11f0a90176c5cf2df9da",
        "topic": "Data centres",
        "answer": "Hosted in US Central.",
    }

    text = load_json(json.dumps(document).encode())[0].text

    assert "53148c64" not in text
    assert "topic: Data centres" in text
    assert "answer: Hosted in US Central." in text


def test_a_meaningful_value_under_an_id_key_is_kept() -> None:
    """Both halves must agree, so a readable value survives an id-ish key."""
    text = load_json(json.dumps({"system_id": "Primary billing cluster"}).encode())[0].text

    assert "system_id: Primary billing cluster" in text


def test_a_hex_looking_value_under_a_meaningful_key_is_kept() -> None:
    text = load_json(json.dumps({"cipher": "aes256cbcdeadbeefdeadbeefdeadbeef"}).encode())[0].text

    assert "aes256cbc" in text
