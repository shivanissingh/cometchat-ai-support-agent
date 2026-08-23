"""
tests/regression/test_bug1_return_window_omission.py — Regression test for Bug 1.

Bug 1: Return Policy Answer Omits the Core Return Window.
Ensures that asking "What are the return policies?" produces an answer and
citations containing the core 30-day return window, item condition, and membership
exception details from 01-returns-policy-current.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agent.orchestrator import ENABLE_SIBLING_BOOST, Agent
from app.session.store import SessionStore


def test_bug1_retrieval_sibling_boost_unit():
    """Unit test (mocked LLM): asserts PATH A sibling boost includes all return policy chunks.

    Verifies that the evidence pack passed to call_llm contains the 'Standard return window'
    section from 01-returns-policy-current.md alongside other sibling sections.
    """
    assert ENABLE_SIBLING_BOOST is True

    store = SessionStore()
    ag = Agent()

    mock_llm = MagicMock(
        return_value=(
            "Mocked response [01-returns-policy-current.md — "
            "Returns Policy > Standard return window]",
            None,
        )
    )

    with (
        patch("app.agent.orchestrator.SESSION_STORE", store),
        patch("app.agent.llm_client.call_llm", mock_llm),
    ):
        response = ag.handle_message("sess-bug1-unit", "What are the return policies?")

    # Assert LLM was called with evidence pack containing the standard return window
    mock_llm.assert_called_once()
    evidence_pack = mock_llm.call_args.kwargs.get("evidence_pack", "")
    assert "01-returns-policy-current.md" in evidence_pack
    assert "Standard return window" in evidence_pack
    assert "30 calendar days" in evidence_pack

    # Assert citations include the standard return window section
    citation_headings = [
        c.get("heading", "").lower()
        for c in response.citations
        if c.get("filename") == "01-returns-policy-current.md"
    ]
    assert any("return window" in h for h in citation_headings)


@pytest.mark.integration
def test_bug1_return_window_omission_live():
    """Live integration test for Bug 1: calls live LLM and verifies answer completeness.

    Asserts:
      a. '30 calendar days' appears in response.answer (case-insensitive).
      b. At least one citation in response.citations has filename '01-returns-policy-current.md'
         and heading containing the words 'return window' (case-insensitive).
      c. '30' appears before '6.95' in response.answer (completeness ordering).
      d. '45' OR 'trailplus' appears in response.answer (case-insensitive).
    """
    store = SessionStore()
    ag = Agent()

    with patch("app.agent.orchestrator.SESSION_STORE", store):
        response = ag.handle_message("sess-bug1-live", "What are the return policies?")

    ans_lower = response.answer.lower()

    # a. '30 calendar days' appears in response.answer (case-insensitive)
    assert "30 calendar days" in ans_lower, (
        f"'30 calendar days' missing from answer: {response.answer}"
    )

    # b. At least one citation from 01-returns-policy-current.md with return window in heading
    has_window_citation = any(
        c.get("filename") == "01-returns-policy-current.md"
        and "return window" in c.get("heading", "").lower()
        for c in response.citations
    )
    assert has_window_citation, (
        f"Missing citation for 01-returns-policy-current.md return window in {response.citations}"
    )

    # c. '30' appears before '6.95' in response.answer
    idx_30 = response.answer.find("30")
    idx_695 = response.answer.find("6.95")
    assert idx_30 != -1, "'30' not found in answer"
    assert idx_695 != -1, "'6.95' not found in answer"
    assert idx_30 < idx_695, (
        f"'30' (pos {idx_30}) must appear before '6.95' (pos {idx_695}) in answer: "
        f"{response.answer}"
    )

    # d. '45' OR 'trailplus' appears in response.answer (case-insensitive)
    assert "45" in ans_lower or "trailplus" in ans_lower, (
        f"Neither '45' nor 'trailplus' found in answer: {response.answer}"
    )
