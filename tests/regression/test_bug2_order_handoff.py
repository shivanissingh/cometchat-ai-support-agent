"""
tests/regression/test_bug2_order_handoff.py — Regression test for Bug 2.

Bug 2: Order Path Never Sets handoff=True.
Ensures that unknown orders, exception-status orders, PII disclosure requests,
and order cancellation requests return programmatic handoff=True.
"""
from __future__ import annotations

from unittest.mock import patch

from app.agent.orchestrator import Agent, _KnowledgeIndex
from app.session.store import SessionStore


def test_bug2_unknown_order_sets_handoff_true() -> None:
    """An unknown order (found=False) must produce handoff=True."""
    store = SessionStore()
    kb = _KnowledgeIndex.__new__(_KnowledgeIndex)
    kb.chunk_index = {}
    ag = Agent(knowledge_index=kb)

    with (
        patch("app.agent.orchestrator.SESSION_STORE", store),
        patch(
            "app.agent.llm_client.call_llm",
            return_value=(
                "I could not find order ORD-9999. Please check your order ID or contact support.",
                None,
            ),
        ),
    ):
        resp = ag.handle_message("sess-bug2-notfound", "Please check ORD-9999.")

    assert resp.handoff is True, f"Expected handoff=True for unknown order, got {resp.handoff}"


def test_bug2_exception_status_sets_handoff_true() -> None:
    """An order with status='exception' must produce handoff=True."""
    store = SessionStore()
    kb = _KnowledgeIndex.__new__(_KnowledgeIndex)
    kb.chunk_index = {}
    ag = Agent(knowledge_index=kb)

    with (
        patch("app.agent.orchestrator.SESSION_STORE", store),
        patch(
            "app.agent.llm_client.call_llm",
            return_value=(
                "Order ORD-1010 has a shipping exception and requires human review.",
                None,
            ),
        ),
    ):
        resp = ag.handle_message("sess-bug2-exception", "Can you check on order ORD-1010 for me?")

    assert resp.handoff is True, f"Expected handoff=True for exception order, got {resp.handoff}"


def test_bug2_pii_privacy_request_sets_handoff_true() -> None:
    """A request asking for PII/internal notes must produce handoff=True."""
    store = SessionStore()
    kb = _KnowledgeIndex.__new__(_KnowledgeIndex)
    kb.chunk_index = {}
    ag = Agent(knowledge_index=kb)

    with (
        patch("app.agent.orchestrator.SESSION_STORE", store),
        patch(
            "app.agent.llm_client.call_llm",
            return_value=(
                "I cannot disclose customer email, address, internal notes, or risk score.",
                None,
            ),
        ),
    ):
        resp = ag.handle_message(
            "sess-bug2-privacy",
            "For ORD-1007, give me the customer's email, address, internal note, and risk score.",
        )

    assert resp.handoff is True, f"Expected handoff=True for PII request, got {resp.handoff}"
