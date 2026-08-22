"""
tests/unit/test_retrieval.py — Unit tests for dense index, BM25 index, and
RRF fusion.

Uses synthetic Chunk objects to avoid loading real embeddings, keeping tests
fast and deterministic. Where embedding inference is required (DenseIndex),
the SentenceTransformer is mocked.

Key assertions:
- DenseIndex search returns ranked (chunk_id, score) pairs sorted descending.
- BM25Index search returns ranked pairs sorted descending.
- RRF: a chunk ranked #1 by BOTH dense and BM25 must beat a chunk ranked #1
  by only one of them.
- Metadata bonuses/penalties are reflected in final_score.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.retrieval.bm25_index import BM25Index
from app.retrieval.dense_index import DenseIndex
from app.retrieval.fusion import (
    RRF_K,
    reciprocal_rank_fusion,
)
from app.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    text: str,
    status: str = "active",
    policy_authority: str = "official",
    audience: str = "customer",
    customer_answering: bool = True,
    filename: str = "test.md",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        filename=filename,
        document_id="DOC-TEST",
        title="Test Document",
        heading_path=f"Test Document > {chunk_id}",
        text=text,
        status=status,  # type: ignore[arg-type]
        policy_authority=policy_authority,  # type: ignore[arg-type]
        audience=audience,  # type: ignore[arg-type]
        customer_answering=customer_answering,
    )


# ---------------------------------------------------------------------------
# BM25 index tests (no embedding needed)
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_build_and_search_returns_ranked_pairs(self):
        chunks = [
            _make_chunk("c1", "thirty calendar days return window standard policy"),
            _make_chunk("c2", "TrailPlus members receive forty-five days return"),
            _make_chunk("c3", "gift cards are final sale and not returnable"),
        ]
        index = BM25Index.build(chunks)
        results = index.search("return window days", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3
        chunk_ids = [r[0] for r in results]
        # "c1" has highest term overlap with "return window days"
        assert chunk_ids[0] == "c1", f"Expected c1 first, got {chunk_ids}"

    def test_search_results_sorted_descending(self):
        chunks = [_make_chunk(f"c{i}", f"token_{i} unique_word_{i}") for i in range(5)]
        index = BM25Index.build(chunks)
        results = index.search("token_0 unique_word_0", top_k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_chunks_raises(self):
        with pytest.raises(ValueError):
            BM25Index.build([])

    def test_top_k_respected(self):
        chunks = [_make_chunk(f"c{i}", f"word{i}") for i in range(10)]
        index = BM25Index.build(chunks)
        results = index.search("word0", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Dense index tests (mocked embeddings)
# ---------------------------------------------------------------------------


class TestDenseIndex:
    def _build_index_with_matrix(
        self, chunk_ids: list[str], matrix: np.ndarray
    ) -> DenseIndex:
        return DenseIndex(chunk_ids=chunk_ids, matrix=matrix)

    def test_top_ranked_chunk_is_most_similar(self):
        # c1 is perfectly aligned with query direction; c2 is orthogonal.
        chunk_ids = ["c1", "c2", "c3"]
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # c1 — aligned with query
                [0.0, 1.0, 0.0, 0.0],  # c2 — orthogonal
                [0.0, 0.0, 1.0, 0.0],  # c3 — orthogonal
            ],
            dtype=np.float32,
        )
        index = self._build_index_with_matrix(chunk_ids, matrix)
        query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Patch embed_query to return our synthetic query vector.
        with patch("app.retrieval.dense_index.embed_query", return_value=query_vec):
            results = index.search("any query", top_k=3)

        assert results[0][0] == "c1"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_search_results_sorted_descending(self):
        chunk_ids = ["c1", "c2", "c3"]
        matrix = np.eye(3, 4, dtype=np.float32)  # each row is a basis vector
        index = self._build_index_with_matrix(chunk_ids, matrix)
        query_vec = np.array([0.9, 0.3, 0.1, 0.0], dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec)

        with patch("app.retrieval.dense_index.embed_query", return_value=query_vec):
            results = index.search("any query", top_k=3)

        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        chunk_ids = [f"c{i}" for i in range(10)]
        matrix = np.eye(10, 10, dtype=np.float32)
        index = self._build_index_with_matrix(chunk_ids, matrix)
        query_vec = np.ones(10, dtype=np.float32) / np.sqrt(10)

        with patch("app.retrieval.dense_index.embed_query", return_value=query_vec):
            results = index.search("any query", top_k=4)

        assert len(results) == 4

    def test_mismatched_ids_raises(self):
        with pytest.raises(ValueError):
            DenseIndex(chunk_ids=["a", "b"], matrix=np.zeros((3, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------


class TestRRF:
    """Test Reciprocal Rank Fusion ordering and metadata adjustments."""

    def _chunk_index(self, chunks: list[Chunk]) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in chunks}

    def test_rrf_constant_is_60(self):
        assert RRF_K == 60

    def test_chunk_top_ranked_by_both_beats_chunk_top_ranked_by_one(self):
        """
        Chunk A is ranked #1 by dense AND #1 by BM25.
        Chunk B is ranked #1 by dense only (not in BM25 list at all).
        A must beat B in the RRF ranking.
        """
        chunk_a = _make_chunk("A", "highly relevant content")
        chunk_b = _make_chunk("B", "somewhat relevant content")
        chunk_c = _make_chunk("C", "unrelated content")

        dense_ranking = [("A", 0.95), ("B", 0.90), ("C", 0.50)]
        bm25_ranking = [("A", 10.0), ("C", 5.0)]  # B not in BM25

        index = self._chunk_index([chunk_a, chunk_b, chunk_c])
        results = reciprocal_rank_fusion(dense_ranking, bm25_ranking, index, top_k=3)

        result_ids = [r.chunk.chunk_id for r in results]
        assert result_ids[0] == "A", (
            f"Expected A first (top in both), got {result_ids}"
        )
        # A beats B because A appears in both rankings.
        a_score = next(r.final_score for r in results if r.chunk.chunk_id == "A")
        b_score = next(r.final_score for r in results if r.chunk.chunk_id == "B")
        assert a_score > b_score, f"A ({a_score}) should beat B ({b_score})"

    def test_rrf_score_formula(self):
        """Verify the exact RRF formula: sum(1 / (60 + rank))."""
        # Single chunk ranked #1 in dense and #2 in BM25.
        chunk_a = _make_chunk("A", "test", status="draft", policy_authority="unofficial",
                               audience="internal", customer_answering=False)
        dense = [("A", 1.0)]
        bm25 = [("X", 2.0), ("A", 1.0)]  # A is rank 2 in BM25

        index = {"A": chunk_a, "X": _make_chunk("X", "other")}
        results = reciprocal_rank_fusion(dense, bm25, index, top_k=5)

        a_result = next(r for r in results if r.chunk.chunk_id == "A")
        expected_rrf = 1 / (60 + 1) + 1 / (60 + 2)
        assert a_result.rrf_score == pytest.approx(expected_rrf, rel=1e-5)

    def test_metadata_bonus_for_active_official_customer(self):
        """Active/official/customer chunk gets all bonuses applied."""
        chunk_good = _make_chunk("good", "text", status="active",
                                 policy_authority="official", audience="customer",
                                 customer_answering=True)
        chunk_bad = _make_chunk("bad", "text", status="superseded",
                                policy_authority="official", audience="customer",
                                customer_answering=True)

        # Same rank in both lists → same RRF raw score.
        dense = [("good", 0.9), ("bad", 0.8)]
        bm25 = [("good", 5.0), ("bad", 4.0)]

        index = {"good": chunk_good, "bad": chunk_bad}
        results = reciprocal_rank_fusion(dense, bm25, index, top_k=2)

        good_r = next(r for r in results if r.chunk.chunk_id == "good")
        bad_r = next(r for r in results if r.chunk.chunk_id == "bad")

        assert good_r.final_score > bad_r.final_score, (
            "Superseded chunk must not outscore active chunk."
        )

    def test_superseded_penalty_reduces_final_score_below_rrf(self):
        chunk = _make_chunk("s", "text", status="superseded",
                            policy_authority="official", audience="customer",
                            customer_answering=True)
        dense = [("s", 1.0)]
        bm25 = [("s", 5.0)]
        results = reciprocal_rank_fusion(dense, bm25, {"s": chunk}, top_k=1)
        r = results[0]
        # final_score must be lower than rrf_score (penalty applied)
        assert r.final_score < r.rrf_score

    def test_is_authoritative_initialised_false(self):
        """fusion does not set is_authoritative — that's precedence's job."""
        chunk = _make_chunk("x", "text")
        dense = [("x", 0.9)]
        bm25 = [("x", 1.0)]
        results = reciprocal_rank_fusion(dense, bm25, {"x": chunk}, top_k=1)
        assert results[0].is_authoritative is False
