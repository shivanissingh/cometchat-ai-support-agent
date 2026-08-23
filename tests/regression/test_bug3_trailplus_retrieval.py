"""
tests/regression/test_bug3_trailplus_retrieval.py — Regression test for Bug 3.

Bug 3: Retrieval Miss — TrailPlus Return Window Not Surfaced.
Ensures that queries mentioning TrailPlus force 09-trailplus-membership.md into the evidence pack.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agent.orchestrator import Agent
from app.session.store import SessionStore


def test_bug3_trailplus_forced_evidence_pack_unit() -> None:
    """Asserts that a TrailPlus query includes 09-trailplus-membership.md chunks."""
    store = SessionStore()
    ag = Agent()

    mock_llm = MagicMock(
        return_value=(
            "TrailPlus members receive 45 calendar days to return items. "
            "[09-trailplus-membership.md — Return window]",
            None,
        )
    )

    with (
        patch("app.agent.orchestrator.SESSION_STORE", store),
        patch("app.agent.llm_client.call_llm", mock_llm),
    ):
        response = ag.handle_message(
            "sess-bug3-unit",
            "My TrailPlus membership was active when I ordered. What is my return window?",
        )

    mock_llm.assert_called_once()
    evidence_pack = mock_llm.call_args.kwargs.get("evidence_pack", "")
    assert "09-trailplus-membership.md" in evidence_pack, (
        f"09-trailplus-membership.md missing from evidence pack: {evidence_pack}"
    )

    # Verify 09-trailplus-membership.md is cited in citations list
    citation_files = [c.get("filename") for c in response.citations]
    assert "09-trailplus-membership.md" in citation_files, (
        f"09-trailplus-membership.md missing from citations: {response.citations}"
    )
