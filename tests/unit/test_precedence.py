"""
tests/unit/test_precedence.py — Unit tests for the precedence module.

Verifies that mark_authoritative() correctly distinguishes authoritative
chunks (active + official + customer) from non-authoritative ones.

Key scenario:
  Given the "return policy" query, the superseded legacy 30-day doc (status=
  superseded) must NOT be marked authoritative, while the current 30-day doc
  (status=active) must be.

  This prevents the pipeline from ever surfacing the legacy policy as a source
  of truth for a customer, even if it scores well in retrieval.
"""

from __future__ import annotations

from app.policy.precedence import mark_authoritative
from app.schemas import Chunk, RetrievedChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
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
        title="Test",
        heading_path=f"Test > {chunk_id}",
        text="placeholder text",
        status=status,  # type: ignore[arg-type]
        policy_authority=policy_authority,  # type: ignore[arg-type]
        audience=audience,  # type: ignore[arg-type]
        customer_answering=customer_answering,
    )


def _make_retrieved(chunk: Chunk, final_score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk,
        dense_score=0.5,
        bm25_score=0.5,
        rrf_score=0.5,
        final_score=final_score,
        is_authoritative=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMarkAuthoritative:
    def test_active_official_customer_is_authoritative(self):
        chunk = _make_chunk(
            "current", status="active", policy_authority="official", audience="customer"
        )
        result = mark_authoritative([_make_retrieved(chunk)])
        assert result[0].is_authoritative is True

    def test_superseded_is_not_authoritative(self):
        """The legacy returns policy (status=superseded) must never be authoritative."""
        chunk = _make_chunk(
            "legacy", status="superseded", policy_authority="official", audience="customer"
        )
        result = mark_authoritative([_make_retrieved(chunk)])
        assert result[0].is_authoritative is False

    def test_draft_is_not_authoritative(self):
        chunk = _make_chunk(
            "draft", status="draft", policy_authority="official", audience="customer"
        )
        result = mark_authoritative([_make_retrieved(chunk)])
        assert result[0].is_authoritative is False

    def test_internal_audience_is_not_authoritative(self):
        """13-support-escalation.md is audience=internal — must not be authoritative."""
        chunk = _make_chunk(
            "escalation", status="active", policy_authority="official", audience="internal"
        )
        result = mark_authoritative([_make_retrieved(chunk)])
        assert result[0].is_authoritative is False

    def test_unofficial_is_not_authoritative(self):
        chunk = _make_chunk(
            "unofficial", status="active", policy_authority="unofficial", audience="customer"
        )
        result = mark_authoritative([_make_retrieved(chunk)])
        assert result[0].is_authoritative is False

    def test_none_authority_is_not_authoritative(self):
        """14-internal-content-migration-notes.md has policy_authority=none."""
        chunk = _make_chunk(
            "scratchpad", status="draft", policy_authority="none", audience="internal"
        )
        result = mark_authoritative([_make_retrieved(chunk)])
        assert result[0].is_authoritative is False

    def test_mixed_list_marks_correctly(self):
        current = _make_chunk(
            "current", status="active", policy_authority="official", audience="customer"
        )
        legacy = _make_chunk(
            "legacy", status="superseded", policy_authority="official", audience="customer"
        )
        internal = _make_chunk(
            "internal", status="active", policy_authority="official", audience="internal"
        )
        trailplus = _make_chunk(
            "trailplus", status="active", policy_authority="official", audience="customer"
        )

        chunks = [current, legacy, internal, trailplus]
        results = mark_authoritative([_make_retrieved(c) for c in chunks])

        auth_map = {r.chunk.chunk_id: r.is_authoritative for r in results}
        assert auth_map["current"] is True
        assert auth_map["legacy"] is False
        assert auth_map["internal"] is False
        assert auth_map["trailplus"] is True

    def test_return_policy_scenario(self):
        """End-to-end precedence: current policy is authoritative, legacy is not."""
        from pathlib import Path

        from app.ingestion.chunker import chunk_document
        from app.ingestion.parser import parse_file

        kb = Path(__file__).parent.parent.parent / "knowledge-base"
        current_doc = parse_file(kb / "01-returns-policy-current.md")
        legacy_doc = parse_file(kb / "02-returns-policy-legacy.md")

        current_chunks = chunk_document(current_doc)
        legacy_chunks = chunk_document(legacy_doc)

        all_retrieved = [_make_retrieved(c, final_score=0.8) for c in current_chunks] + \
                        [_make_retrieved(c, final_score=0.9) for c in legacy_chunks]

        results = mark_authoritative(all_retrieved)

        for r in results:
            if r.chunk.filename == "01-returns-policy-current.md":
                assert r.is_authoritative is True, "Current policy must be authoritative"
            elif r.chunk.filename == "02-returns-policy-legacy.md":
                assert r.is_authoritative is False, "Legacy policy must NOT be authoritative"

    def test_empty_list(self):
        assert mark_authoritative([]) == []

    def test_original_list_not_mutated(self):
        """mark_authoritative must not modify chunks in-place."""
        chunk = _make_chunk("x", status="active", policy_authority="official", audience="customer")
        rc = _make_retrieved(chunk)
        original_flag = rc.is_authoritative
        mark_authoritative([rc])
        # The original RetrievedChunk should not have been mutated.
        assert rc.is_authoritative == original_flag
