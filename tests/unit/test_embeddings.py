"""
tests/unit/test_embeddings.py — Unit tests for the embedding wrapper.

Critical invariant (MUST NOT be silently dropped):
    For BAAI/bge-small-en-v1.5, the BGE instruction prefix is prepended to
    QUERY text and is NEVER prepended to PASSAGE text.

These tests are intentionally written to fail loudly if the prefix logic
is removed, restructured, or silently short-circuited.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.retrieval.embeddings import (
    BGE_QUERY_PREFIX,
    embed_passages,
    embed_query,
    is_bge_model,
)

# ---------------------------------------------------------------------------
# Model-name classification tests
# ---------------------------------------------------------------------------


class TestIsBlgModel:
    def test_bge_small_en_identified_as_bge(self):
        assert is_bge_model("BAAI/bge-small-en-v1.5") is True

    def test_bge_large_identified_as_bge(self):
        assert is_bge_model("BAAI/bge-large-en-v1.5") is True

    def test_case_insensitive(self):
        assert is_bge_model("baai/bge-small-en-v1.5") is True

    def test_minilm_not_bge(self):
        assert is_bge_model("all-MiniLM-L6-v2") is False

    def test_other_model_not_bge(self):
        assert is_bge_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") is False


# ---------------------------------------------------------------------------
# BGE prefix application tests
# ---------------------------------------------------------------------------


class TestBGEPrefixLogic:
    """These tests use a mock SentenceTransformer to avoid downloading models."""

    def _make_mock_model(self, dim: int = 8) -> MagicMock:
        """Return a mock that returns a float32 unit vector for any encode() call."""
        mock = MagicMock()
        vec = np.ones(dim, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        mock.encode.return_value = vec
        return mock

    def test_bge_query_receives_prefix(self):
        """embed_query with a BGE model must prepend the instruction prefix."""
        mock_model = self._make_mock_model()

        with (
            patch("app.retrieval.embeddings._get_model", return_value=mock_model),
            patch("app.retrieval.embeddings.is_bge_model", return_value=True),
        ):
            embed_query("what is the return policy?", model_name="BAAI/bge-small-en-v1.5")

        call_args = mock_model.encode.call_args
        text_passed = call_args[0][0]
        assert text_passed.startswith(BGE_QUERY_PREFIX), (
            f"BGE query must start with prefix. Got: {text_passed!r}"
        )

    def test_bge_passage_does_not_receive_prefix(self):
        """embed_passages with a BGE model must NOT prepend any prefix."""
        mock_model = self._make_mock_model()
        mat = np.ones((2, 8), dtype=np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        mock_model.encode.return_value = mat

        passages = ["Customers may return items within 30 days.", "TrailPlus members get 45 days."]

        with (
            patch("app.retrieval.embeddings._get_model", return_value=mock_model),
            patch("app.retrieval.embeddings.is_bge_model", return_value=True),
        ):
            embed_passages(passages, model_name="BAAI/bge-small-en-v1.5")

        call_args = mock_model.encode.call_args
        texts_passed = call_args[0][0]  # first positional arg
        # Passages must be passed as-is — no prefix on any of them.
        for p, t in zip(passages, texts_passed):
            assert t == p, f"Passage was modified before embedding. Expected {p!r}, got {t!r}"

    def test_minilm_query_receives_no_prefix(self):
        """embed_query with all-MiniLM-L6-v2 must NOT prepend any prefix."""
        mock_model = self._make_mock_model()
        query = "what is the return policy?"

        with (
            patch("app.retrieval.embeddings._get_model", return_value=mock_model),
            patch("app.retrieval.embeddings.is_bge_model", return_value=False),
        ):
            embed_query(query, model_name="all-MiniLM-L6-v2")

        call_args = mock_model.encode.call_args
        text_passed = call_args[0][0]
        assert text_passed == query, f"MiniLM query must not be modified. Got: {text_passed!r}"

    def test_prefix_is_never_empty_string(self):
        """BGE_QUERY_PREFIX must be a non-empty, meaningful string."""
        assert len(BGE_QUERY_PREFIX.strip()) > 10, (
            "BGE_QUERY_PREFIX appears to be empty or trivially short — "
            "this may indicate it was silently dropped."
        )

    def test_embed_passages_returns_2d_array(self):
        """embed_passages output shape must be (n_texts, dim)."""
        mock_model = self._make_mock_model(dim=16)
        mat = np.ones((3, 16), dtype=np.float32)
        mock_model.encode.return_value = mat

        with patch("app.retrieval.embeddings._get_model", return_value=mock_model):
            result = embed_passages(["a", "b", "c"])

        assert result.ndim == 2
        assert result.shape[0] == 3

    def test_embed_query_returns_1d_array(self):
        """embed_query output shape must be (dim,)."""
        mock_model = self._make_mock_model(dim=16)

        with (
            patch("app.retrieval.embeddings._get_model", return_value=mock_model),
            patch("app.retrieval.embeddings.is_bge_model", return_value=False),
        ):
            result = embed_query("hello")

        assert result.ndim == 1

    def test_embed_passages_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            embed_passages([])
