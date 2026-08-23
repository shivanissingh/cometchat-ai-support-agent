"""
app/retrieval/bm25_index.py — BM25 sparse retrieval index.

Uses rank_bm25.BM25Okapi with simple lowercase/whitespace tokenisation.
This is intentionally minimal — no stemming, stopword removal, or
subword tokenisation — which is sufficient for a small, well-curated
knowledge base like this one.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase and split *text* on non-alphanumeric characters."""
    return re.split(r"\W+", text.lower().strip())


# ---------------------------------------------------------------------------
# Index class
# ---------------------------------------------------------------------------


class BM25Index:
    """BM25Okapi index over a fixed set of chunk texts.

    Parameters
    ----------
    chunk_ids:
        Ordered list of chunk_id strings parallel to *texts*.
    texts:
        Passage texts to index (one per chunk_id).
    """

    def __init__(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError(
                f"chunk_ids length {len(chunk_ids)} does not match texts length {len(texts)}"
            )
        self._chunk_ids: list[str] = chunk_ids
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, chunks: list) -> BM25Index:  # chunks: list[Chunk]
        """Build a BM25Index from a list of Chunk objects."""
        if not chunks:
            raise ValueError("Cannot build a BM25Index from an empty chunk list.")
        chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        return cls(chunk_ids=chunk_ids, texts=texts)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the top-*k* (chunk_id, bm25_score) pairs for *query*.

        Parameters
        ----------
        query:
            Raw query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            Pairs sorted by descending BM25 score.  Pairs with score == 0.0
            are included only up to top_k; callers may filter them out.
        """
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)  # numpy array, one per chunk

        top_k = min(top_k, len(self._chunk_ids))
        # argsort ascending → flip for descending
        import numpy as np

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self._chunk_ids[int(i)], float(scores[int(i)])) for i in top_indices]

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)
