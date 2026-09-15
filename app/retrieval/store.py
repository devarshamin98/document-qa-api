"""In-memory faiss vector store implementing the `VectorStore` port."""

from collections.abc import Sequence

import faiss
import numpy as np

from app.core.ports import Chunk, ScoredChunk


class FaissStore:
    """Cosine similarity over an `IndexFlatIP` of L2-normalized vectors.

    One instance per request: the index is built from the uploaded document and
    discarded with it. Exact search, so there is no recall/accuracy tradeoff to
    reason about at document scale.

    Swapping faiss for plain numpy would be a change confined to this class.
    """

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors; they must correspond."
            )

        matrix = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        self._index = index
        self._chunks = list(chunks)

    async def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        """Top-k by cosine similarity, highest first.

        Async to satisfy the port; the work itself is microsecond-scale CPU on an
        exact flat index, so it does not warrant a threadpool hop.
        """
        if self._index is None or not self._chunks:
            return []

        query = np.asarray([vector], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(k, len(self._chunks)))

        hits: list[ScoredChunk] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:  # faiss pads with -1 when fewer than k results exist
                continue
            hits.append(ScoredChunk(chunk=self._chunks[int(index)], score=float(score)))
        return hits
