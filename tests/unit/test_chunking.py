"""Chunking: overlap, tiny documents, and page metadata."""

from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import Page


def test_document_smaller_than_one_chunk_stays_whole() -> None:
    pages = [Page(number=1, text="Acme relies on AWS.")]
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].text == "Acme relies on AWS."
    assert chunks[0].id == 1
    assert chunks[0].page == 1


def test_long_text_splits_with_overlapping_content() -> None:
    text = " ".join(f"w{index:03d}" for index in range(200))
    chunks = chunk_pages([Page(number=1, text=text)], chunk_size=100, chunk_overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 100

    # Overlap means adjacent chunks share content; without it retrieval loses
    # anything that straddles a boundary.
    first, second = set(chunks[0].text.split()), set(chunks[1].text.split())
    assert first & second


def test_chunk_ids_are_sequential_across_pages_and_pages_are_preserved() -> None:
    pages = [
        Page(number=1, text="alpha " * 100),
        Page(number=2, text="beta " * 100),
        Page(number=7, text="gamma " * 100),
    ]
    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=20)

    assert [chunk.id for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert {chunk.page for chunk in chunks} == {1, 2, 7}
    # A chunk never straddles a page boundary, so citations stay honest.
    assert all("beta" not in chunk.text for chunk in chunks if chunk.page == 1)


def test_pages_with_no_usable_text_produce_no_chunks() -> None:
    assert chunk_pages([Page(number=1, text="   ")], chunk_size=100, chunk_overlap=10) == []
