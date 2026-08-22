"""
tests/integration/conftest.py — Shared pytest fixtures for integration tests.

Fixtures
--------
kb_index:
    A _KnowledgeIndex built once per test session (session scope).  Heavy
    to build; shared to keep the suite fast.

agent_factory:
    Returns a factory function that creates an Agent pre-wired to an
    isolated SessionStore.  The factory does NOT patch call_llm — each
    test owns its own LLM patch so there is no double-patch conflict.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agent.orchestrator import Agent, _KnowledgeIndex
from app.session.store import SessionStore


@pytest.fixture(scope="session")
def kb_index() -> _KnowledgeIndex:
    """Build the knowledge-base index once per test session."""
    return _KnowledgeIndex()


@pytest.fixture()
def agent_factory(kb_index: _KnowledgeIndex):  # noqa: ANN201
    """Factory that creates an Agent with an isolated SessionStore.

    Each test that needs a mocked LLM must apply its own
    ``patch("app.agent.llm_client.call_llm", ...)`` context manager.

    Usage::

        def test_something(agent_factory):
            ag, store = agent_factory()
            with patch("app.agent.llm_client.call_llm", return_value=("answer", None)):
                response = ag.handle_message("s1", "hello")
    """
    patchers: list = []

    def _make():
        store = SessionStore()
        patcher_store = patch("app.agent.orchestrator.SESSION_STORE", store)
        patcher_store.start()
        patchers.append(patcher_store)
        ag = Agent(knowledge_index=kb_index)
        return ag, store

    yield _make

    # Stop all store patches after the test completes.
    for p in patchers:
        try:
            p.stop()
        except RuntimeError:
            pass
