"""
tests/unit/test_conflict.py — Unit tests for ConflictDetector.

Critical test cases
-------------------

TRAP CASE (must NOT fire):
  The TrailPlus 45-day window coexists with the current 30-day standard policy.
  Both documents (01 and 09) are active/official/customer/authoritative.
  The 45-day window is explicitly membership-conditional — the text around it
  contains "TrailPlus" / "membership" — so extract_day_count assigns it to
  claim key "trailplus_member_days", not "standard_days".
  No same-key conflict exists → has_conflict must be False.

REQUIRED CASE (MUST fire):
  11-product-care.md says the Breeze Tumbler body "should be hand-washed".
  12-breeze-tumbler-product-card.md says "all components are dishwasher safe".
  Both are active/official/customer/authoritative.
  extract_dishwasher_claim assigns both to key "body_dishwasher" with different
  values ("hand_wash" vs "dishwasher_safe") → has_conflict must be True.

Additional cases:
  - Synthetic same-standard-day-count contradiction between two active official
    customer docs → conflict fires (proves numeric logic works).
  - Only non-authoritative chunks → no conflict (detector ignores them).
  - Single authoritative chunk per topic → no conflict (nothing to compare).
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.ingestion.parser import parse_file
from app.policy.conflict import ConflictDetector, extract_day_count, extract_dishwasher_claim
from app.schemas import Chunk, RetrievedChunk

KB_DIR = Path(__file__).parent.parent.parent / "knowledge-base"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    text: str,
    topic: str,
    status: str = "active",
    policy_authority: str = "official",
    audience: str = "customer",
    filename: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        filename=filename or f"{chunk_id}.md",
        document_id="SYNTH-01",
        title="Synthetic",
        heading_path=f"Synthetic > {chunk_id}",
        text=text,
        status=status,  # type: ignore[arg-type]
        policy_authority=policy_authority,  # type: ignore[arg-type]
        audience=audience,  # type: ignore[arg-type]
        customer_answering=True,
        topic=topic,
    )


def _auth_retrieved(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk,
        dense_score=0.9,
        bm25_score=5.0,
        rrf_score=0.03,
        final_score=0.25,
        is_authoritative=True,
    )


def _nonauth_retrieved(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk,
        dense_score=0.9,
        bm25_score=5.0,
        rrf_score=0.03,
        final_score=0.05,
        is_authoritative=False,
    )


# ---------------------------------------------------------------------------
# Unit tests for claim extractors
# ---------------------------------------------------------------------------


class TestExtractDayCount:
    def test_extracts_standard_days_from_current_policy(self):
        chunk = _make_chunk(
            "c1",
            "Customers may request a return within 30 calendar days of delivery.",
            "returns",
        )
        claims = extract_day_count(chunk)
        assert claims.get("standard_days") == "30"
        assert "trailplus_member_days" not in claims

    def test_extracts_trailplus_days_as_membership_conditional(self):
        chunk = _make_chunk(
            "c2",
            (
                "A customer whose TrailPlus membership was active when an order was placed "
                "receives a 45-calendar-day return window from delivery for eligible items."
            ),
            "returns",
        )
        claims = extract_day_count(chunk)
        assert claims.get("trailplus_member_days") == "45"
        assert "standard_days" not in claims

    def test_extracts_standard_from_legacy_policy(self):
        chunk = _make_chunk(
            "c3",
            "Customers could return eligible merchandise within 45 calendar days of delivery.",
            "returns",
        )
        claims = extract_day_count(chunk)
        assert claims.get("standard_days") == "45"
        assert "trailplus_member_days" not in claims

    def test_no_days_in_text(self):
        chunk = _make_chunk("c4", "Gift cards are final sale.", "returns")
        assert extract_day_count(chunk) == {}


class TestExtractDishwasherClaim:
    def test_detects_hand_wash_claim(self):
        chunk = _make_chunk(
            "c1",
            "The stainless-steel body of the Breeze Tumbler should be hand-washed.",
            "breeze-tumbler-care",
        )
        claims = extract_dishwasher_claim(chunk)
        assert claims.get("body_dishwasher") == "hand_wash"

    def test_detects_dishwasher_safe_all_components(self):
        chunk = _make_chunk(
            "c2",
            "The product card states that all components are dishwasher safe, "
            "with the top rack recommended.",
            "breeze-tumbler-care",
        )
        claims = extract_dishwasher_claim(chunk)
        assert claims.get("body_dishwasher") == "dishwasher_safe"

    def test_no_claim_for_unrelated_text(self):
        chunk = _make_chunk(
            "c3",
            "Do not microwave any component.",
            "breeze-tumbler-care",
        )
        assert extract_dishwasher_claim(chunk) == {}


# ---------------------------------------------------------------------------
# ConflictDetector integration tests
# ---------------------------------------------------------------------------


class TestConflictDetector:
    def setup_method(self):
        self.detector = ConflictDetector()

    # --- TRAP CASE ---

    def test_trailplus_does_not_conflict_with_current_policy(self):
        """
        TRAP CASE: TrailPlus 45-day exception must NOT trigger a conflict
        against the standard 30-day policy.

        Uses real content from the knowledge base files.
        """
        current_doc = parse_file(KB_DIR / "01-returns-policy-current.md")
        trailplus_doc = parse_file(KB_DIR / "09-trailplus-membership.md")

        current_chunks = chunk_document(current_doc)
        trailplus_chunks = chunk_document(trailplus_doc)

        all_retrieved = [_auth_retrieved(c) for c in current_chunks] + [
            _auth_retrieved(c) for c in trailplus_chunks
        ]

        result = self.detector.detect(all_retrieved)

        assert result.has_conflict is False, (
            "TrailPlus 45-day membership-conditional exception must NOT be "
            f"detected as a conflict with the standard 30-day policy. "
            f"Explanation: {result.explanation}"
        )

    # --- REQUIRED CASE ---

    def test_breeze_tumbler_dishwasher_conflict_detected(self):
        """
        REQUIRED CASE: 11-product-care.md vs 12-breeze-tumbler-product-card.md.

        Both are active/official/customer. One says body must be hand-washed;
        the other says all components are dishwasher safe. This is a genuine
        conflict that MUST be detected.

        Uses real content from the knowledge base files.
        """
        care_doc = parse_file(KB_DIR / "11-product-care.md")
        card_doc = parse_file(KB_DIR / "12-breeze-tumbler-product-card.md")

        care_chunks = chunk_document(care_doc)
        card_chunks = chunk_document(card_doc)

        all_retrieved = [_auth_retrieved(c) for c in care_chunks] + [
            _auth_retrieved(c) for c in card_chunks
        ]

        result = self.detector.detect(all_retrieved)

        assert result.has_conflict is True, (
            "Breeze Tumbler hand-wash vs dishwasher-safe contradiction must "
            "be detected as a conflict."
        )
        assert result.explanation is not None
        assert len(result.conflicting_chunks) >= 2

    def test_breeze_tumbler_explanation_is_customer_friendly(self):
        """Conflict explanation must be a non-empty human-readable string."""
        care_doc = parse_file(KB_DIR / "11-product-care.md")
        card_doc = parse_file(KB_DIR / "12-breeze-tumbler-product-card.md")

        all_retrieved = [_auth_retrieved(c) for c in chunk_document(care_doc)] + [
            _auth_retrieved(c) for c in chunk_document(card_doc)
        ]
        result = self.detector.detect(all_retrieved)

        assert result.has_conflict is True
        assert result.explanation
        assert len(result.explanation) > 20  # must be a real sentence, not a stub

    def test_non_authoritative_chunks_do_not_trigger_conflict(self):
        """
        Superseded legacy policy (is_authoritative=False) must not trigger
        a conflict with the current policy — that is precedence, not conflict.
        """
        current_doc = parse_file(KB_DIR / "01-returns-policy-current.md")
        legacy_doc = parse_file(KB_DIR / "02-returns-policy-legacy.md")

        current_chunks = chunk_document(current_doc)
        legacy_chunks = chunk_document(legacy_doc)

        # Legacy chunks are explicitly non-authoritative (is_authoritative=False).
        all_retrieved = [_auth_retrieved(c) for c in current_chunks] + [
            _nonauth_retrieved(c) for c in legacy_chunks
        ]

        result = self.detector.detect(all_retrieved)

        assert result.has_conflict is False, (
            "A superseded (non-authoritative) doc must not cause a conflict. "
            "Run precedence before conflict detection."
        )

    # --- Synthetic same-claim-key conflict (proves numeric logic) ---

    def test_synthetic_standard_day_conflict(self):
        """
        Two synthetic active/official/customer chunks that both assert different
        standard return windows must be detected as a conflict.

        This tests the numeric logic independently of any ambiguity in the real
        corpus (since the real legacy doc is superseded and non-authoritative).
        """
        chunk_a = _make_chunk(
            "synth_a",
            "Customers may return eligible items within 30 calendar days of delivery.",
            "returns",
            filename="01-returns-policy-current.md",
        )
        chunk_b = _make_chunk(
            "synth_b",
            "Customers may return eligible items within 60 calendar days of delivery.",
            "returns",
            filename="01-returns-policy-current.md",
        )

        # Note: both filenames are "01-returns-policy-current.md" → topic "returns".
        all_retrieved = [_auth_retrieved(chunk_a), _auth_retrieved(chunk_b)]
        result = self.detector.detect(all_retrieved)

        assert result.has_conflict is True, (
            "Two authoritative chunks claiming different standard day counts "
            "must trigger a conflict."
        )

    def test_single_authoritative_chunk_no_conflict(self):
        chunk = _make_chunk(
            "only",
            "Customers may return items within 30 calendar days.",
            "returns",
            filename="01-returns-policy-current.md",
        )
        result = self.detector.detect([_auth_retrieved(chunk)])
        assert result.has_conflict is False

    def test_empty_chunk_list(self):
        result = self.detector.detect([])
        assert result.has_conflict is False

    def test_topic_without_extractor_never_conflicts(self):
        """
        Topics with no registered extractor (e.g. 'warranty') should never
        produce a conflict, even if two authoritative chunks have different text.
        """
        chunk_a = _make_chunk("w1", "Warranty is 1 year.", "warranty", filename="07-warranty.md")
        chunk_b = _make_chunk("w2", "Warranty is 2 years.", "warranty", filename="07-warranty.md")
        result = self.detector.detect([_auth_retrieved(chunk_a), _auth_retrieved(chunk_b)])
        assert result.has_conflict is False
