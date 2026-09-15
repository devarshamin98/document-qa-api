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
    assert pages[0].text == "a.b: 1\nc[0]: 2"


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
