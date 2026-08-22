"""
app/retrieval/embeddings.py — Sentence-transformer embedding wrapper.

Asymmetric model handling
-------------------------
BAAI/bge-small-en-v1.5 is an *asymmetric* retrieval model trained to use a
query-side instruction prefix.  Per the model card, the prefix

    "Represent this sentence for searching relevant passages: "

MUST be prepended to query text only.  Passage text is indexed without any
prefix.  Applying the prefix to passages degrades retrieval quality.

all-MiniLM-L6-v2 is a *symmetric* model; neither queries nor passages use
any prefix.

This distinction is implemented as an explicit, testable branch on the model
name (case-insensitive prefix check for "baai/bge").  The branch is exercised
by a dedicated unit test that will FAIL if the prefix logic is silently dropped.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Instruction prefix required by BGE asymmetric models for query text only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Model-name prefix (lower-cased) used to identify BGE asymmetric models.
_BGE_PREFIX = "baai/bge"


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

_model_instance: SentenceTransformer | None = None


def _get_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Return the cached SentenceTransformer instance, loading it on first call."""
    global _model_instance  # noqa: PLW0603
    if _model_instance is None:
        _model_instance = SentenceTransformer(model_name)
    return _model_instance


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_bge_model(model_name: str = EMBEDDING_MODEL) -> bool:
    """Return True if *model_name* identifies a BGE asymmetric model.

    This function is exposed for unit-test inspection — tests import it to
    verify that the prefix branch is exercised correctly.
    """
    return model_name.lower().startswith(_BGE_PREFIX)


def embed_query(
    text: str,
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """Embed a *query* string, applying the BGE prefix when required.

    Parameters
    ----------
    text:
        The user's query string, without any prefix.
    model_name:
        The model identifier.  Defaults to ``app.config.EMBEDDING_MODEL``.

    Returns
    -------
    numpy.ndarray
        1-D float32 embedding vector.
    """
    model = _get_model(model_name)
    if is_bge_model(model_name):
        text = BGE_QUERY_PREFIX + text
    vector: np.ndarray = model.encode(text, normalize_embeddings=True)
    return vector.astype(np.float32)


def embed_passages(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """Embed a list of *passage* strings without any query prefix.

    Parameters
    ----------
    texts:
        Passage strings to embed (e.g. chunk texts).
    model_name:
        The model identifier.  Defaults to ``app.config.EMBEDDING_MODEL``.

    Returns
    -------
    numpy.ndarray
        2-D float32 matrix of shape (len(texts), embedding_dim).
    """
    if not texts:
        raise ValueError("texts must be a non-empty list")
    model = _get_model(model_name)
    # No prefix — passages are always indexed as-is, regardless of model type.
    matrix: np.ndarray = model.encode(texts, normalize_embeddings=True)
    return matrix.astype(np.float32)
