"""
tests/unit/test_ingestion.py — Unit tests for parser and chunker.

Covers:
- Parser populates all metadata fields correctly from real front matter.
- Parser assigns topic via TOPIC_MAP (not from front matter).
- Parser sets customer_answering=False for the internal scratchpad doc.
- Chunker produces human-readable heading_path values with " > " separator.
- Chunker does not produce empty-text chunks.
- Chunker propagates metadata from ParsedDocument to Chunk.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
from pathlib import Path

from app.ingestion.chunker import HEADING_SEP, chunk_document
from app.ingestion.parser import parse_all, parse_file
from app.policy.topics import TOPIC_MAP

KB_DIR = Path(__file__).parent.parent.parent / "knowledge-base"


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_parse_all_returns_14_documents(self):
        docs = parse_all()
        assert len(docs) == 14

    def test_current_returns_policy_metadata(self):
        doc = parse_file(KB_DIR / "01-returns-policy-current.md")
        assert doc.document_id == "RET-2026-01"
        assert doc.title == "Returns Policy"
        assert doc.status == "active"
        assert doc.policy_authority == "official"
        assert doc.audience == "customer"
        assert doc.customer_answering is True
        assert doc.effective_date == "2026-04-01"
        assert doc.last_reviewed == "2026-07-15"
        assert doc.topic == "returns"

    def test_legacy_returns_policy_is_superseded(self):
        doc = parse_file(KB_DIR / "02-returns-policy-legacy.md")
        assert doc.status == "superseded"
        assert doc.document_id == "RET-2024-01"
        assert doc.topic == "returns"

    def test_trailplus_membership_topic(self):
        doc = parse_file(KB_DIR / "09-trailplus-membership.md")
        assert doc.status == "active"
        assert doc.policy_authority == "official"
        assert doc.topic == "returns"

    def test_product_care_topic(self):
        doc = parse_file(KB_DIR / "11-product-care.md")
        assert doc.topic == "breeze-tumbler-care"

    def test_breeze_tumbler_product_card_topic(self):
        doc = parse_file(KB_DIR / "12-breeze-tumbler-product-card.md")
        assert doc.topic == "breeze-tumbler-care"

    def test_internal_scratchpad_customer_answering_false(self):
        """14-internal-content-migration-notes.md has explicit customer_answering: false."""
        doc = parse_file(KB_DIR / "14-internal-content-migration-notes.md")
        assert doc.customer_answering is False
        assert doc.audience == "internal"
        assert doc.status == "draft"
        assert doc.policy_authority == "none"

    def test_support_escalation_audience_internal(self):
        """13-support-escalation.md is audience=internal despite policy_authority=official."""
        doc = parse_file(KB_DIR / "13-support-escalation.md")
        assert doc.audience == "internal"
        assert doc.policy_authority == "official"
        assert doc.customer_answering is False  # audience != customer

    def test_all_documents_have_valid_status(self):
        docs = parse_all()
        valid = {"active", "superseded", "draft", "internal"}
        for d in docs:
            assert d.status in valid, f"{d.filename}: unexpected status {d.status!r}"

    def test_all_documents_in_topic_map(self):
        docs = parse_all()
        for d in docs:
            assert d.filename in TOPIC_MAP, f"{d.filename} missing from TOPIC_MAP"

    def test_body_not_empty_for_standard_docs(self):
        doc = parse_file(KB_DIR / "01-returns-policy-current.md")
        assert len(doc.body.strip()) > 0


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestChunker:
    def test_heading_path_contains_title_and_heading(self):
        """heading_path must be '<Title> > <H2 heading>'."""
        doc = parse_file(KB_DIR / "01-returns-policy-current.md")
        chunks = chunk_document(doc)
        paths = [c.heading_path for c in chunks]
        # All paths should start with the document title.
        for path in paths:
            assert path.startswith("Returns Policy")
        # At least one path should contain HEADING_SEP.
        assert any(HEADING_SEP in p for p in paths), (
            f"No heading_path contained '{HEADING_SEP}'. Paths: {paths}"
        )

    def test_heading_path_human_readable(self):
        """heading_path must not contain markdown # characters."""
        doc = parse_file(KB_DIR / "05-domestic-shipping.md")
        chunks = chunk_document(doc)
        for c in chunks:
            assert "#" not in c.heading_path, (
                f"heading_path contains '#': {c.heading_path!r}"
            )

    def test_specific_heading_path_value(self):
        """Verify exact heading_path for a known section."""
        doc = parse_file(KB_DIR / "01-returns-policy-current.md")
        chunks = chunk_document(doc)
        expected_path = "Returns Policy > Standard return window"
        matching = [c for c in chunks if c.heading_path == expected_path]
        assert matching, (
            f"Expected chunk with heading_path {expected_path!r} not found. "
            f"Actual paths: {[c.heading_path for c in chunks]}"
        )

    def test_no_empty_text_chunks(self):
        docs = parse_all()
        for doc in docs:
            for chunk in chunk_document(doc):
                assert chunk.text.strip(), (
                    f"Empty chunk in {doc.filename}: {chunk.heading_path!r}"
                )

    def test_metadata_propagated_to_chunks(self):
        doc = parse_file(KB_DIR / "01-returns-policy-current.md")
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.filename == "01-returns-policy-current.md"
            assert c.document_id == "RET-2026-01"
            assert c.status == "active"
            assert c.policy_authority == "official"
            assert c.audience == "customer"
            assert c.customer_answering is True
            assert c.topic == "returns"

    def test_legacy_doc_chunks_are_superseded(self):
        doc = parse_file(KB_DIR / "02-returns-policy-legacy.md")
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.status == "superseded"

    def test_internal_doc_chunks_have_no_customer_answering(self):
        doc = parse_file(KB_DIR / "14-internal-content-migration-notes.md")
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.customer_answering is False
            assert c.audience == "internal"

    def test_chunk_ids_are_unique(self):
        from app.ingestion.chunker import chunk_all_documents
        from app.ingestion.parser import parse_all as _parse_all

        docs = _parse_all()
        chunks = chunk_all_documents(docs)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"

    def test_chunk_ids_contain_document_id(self):
        doc = parse_file(KB_DIR / "01-returns-policy-current.md")
        chunks = chunk_document(doc)
        for c in chunks:
            assert "RET-2026-01" in c.chunk_id
