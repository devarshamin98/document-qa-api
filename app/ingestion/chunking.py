"""Split pages into overlapping chunks, preserving page numbers."""

from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.ports import Chunk
from app.ingestion.loaders import Page


def chunk_pages(pages: Sequence[Page], *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk each page separately so every chunk carries a real page number.

    Splitting per page means a chunk never straddles a page boundary, which keeps
    citations honest at the cost of a few short chunks at page ends.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Chunk] = []
    next_id = 1
    for page in pages:
        for piece in splitter.split_text(page.text):
            text = piece.strip()
            if not text:
                continue
            chunks.append(Chunk(id=next_id, text=text, page=page.number))
            next_id += 1
    return chunks
