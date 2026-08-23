"""
app/retrieval/dense_index.py — NumPy-based dense vector index.

Builds a float32 matrix of chunk embeddings at index-build time and performs
cosine similarity search at query time.  No FAISS or other vector DB library
is used.  The embeddings matrix fits comfortably in memory for a corpus of
this size (≤ a few hundred chunks × 384 dims).
"""

from __future__ import annotations

import numpy as np

from app.retrieval.embeddings import embed_passages, embed_query

# ---------------------------------------------------------------------------
# Index class
# ---------------------------------------------------------------------------


class DenseIndex:
    """In-memory dense retrieval index backed by a NumPy matrix.

    Parameters
    ----------
    chunk_ids:
        Ordered list of chunk_id strings corresponding to rows in the matrix.
    matrix:
        2-D float32 numpy array of shape (n_chunks, embedding_dim).
        Rows must already be L2-normalised (unit vectors).

    Notes
    -----
    Since the vectors are L2-normalised, cosine similarity reduces to a
    simple dot product: ``similarity = matrix @ query_vec``.
    """

    def __init__(self, chunk_ids: list[str], matrix: np.ndarray) -> None:
        if len(chunk_ids) != matrix.shape[0]:
            raise ValueError(
                f"chunk_ids length {len(chunk_ids)} does not match matrix rows {matrix.shape[0]}"
            )
        self._chunk_ids: list[str] = chunk_ids
        self._matrix: np.ndarray = matrix.astype(np.float32)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, chunks: list) -> DenseIndex:  # chunks: list[Chunk]
        """Build a DenseIndex from a list of Chunk objects.

        Parameters
        ----------
        chunks:
            List of ``app.schemas.Chunk`` instances.  The ``.text`` attribute
            is used as the passage text.

        Returns
        -------
        DenseIndex
            Ready-to-search index.
        """
        if not chunks:
            raise ValueError("Cannot build a DenseIndex from an empty chunk list.")
        chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        matrix = embed_passages(texts)  # already normalised
        return cls(chunk_ids=chunk_ids, matrix=matrix)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the top-*k* (chunk_id, cosine_score) pairs for *query*.

        Parameters
        ----------
        query:
            Raw query string (no prefix; embed_query handles prefix logic).
        top_k:
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            Pairs of (chunk_id, score) sorted by descending cosine similarity.
        """
        query_vec = embed_query(query)  # shape (dim,), already normalised
        scores: np.ndarray = self._matrix @ query_vec  # (n_chunks,)

        top_k = min(top_k, len(self._chunk_ids))
        # argsort ascending → flip for descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self._chunk_ids[int(i)], float(scores[int(i)])) for i in top_indices]

    # ------------------------------------------------------------------
    # Accessors (useful for tests)
    # ------------------------------------------------------------------

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix
