"""
tests/unit/test_session_cross_isolation.py

Tests that two session_ids through the same Agent process never share state.
"""
from __future__ import annotations

from app.session.store import SessionStore, Turn


def _make_turn(order_id=None, topic=None):
    return Turn(user_msg="q", agent_response="a", order_id=order_id, topic=topic)


class TestSessionStoreCrossIsolation:
    """Unit tests for cross-session isolation in SessionStore."""

    def test_two_sessions_independent_turns(self) -> None:
        """Two session IDs must never share turns."""
        store = SessionStore()
        store.add_turn("sess-A", Turn(
            user_msg="Hello", agent_response="Hi A", order_id=None, topic=None
        ))
        store.add_turn("sess-B", Turn(
            user_msg="World", agent_response="Hi B", order_id=None, topic=None
        ))

        turns_a = store.get_turns("sess-A")
        turns_b = store.get_turns("sess-B")

        assert len(turns_a) == 1
        assert len(turns_b) == 1
        assert turns_a[0].agent_response == "Hi A"
        assert turns_b[0].agent_response == "Hi B"

    def test_order_id_isolation(self) -> None:
        """Order ID set in session A must not appear in session B."""
        store = SessionStore()
        store.add_turn("sess-A", _make_turn(order_id="ORD-1005"))
        store.add_turn("sess-B", _make_turn(order_id="ORD-1007"))

        assert store.get_last_order_id("sess-A") == "ORD-1005"
        assert store.get_last_order_id("sess-B") == "ORD-1007"
        assert (
            store.get_last_order_id("sess-A") != store.get_last_order_id("sess-B")
        )

    def test_unknown_session_returns_empty(self) -> None:
        """A session that was never used returns an empty list and None IDs."""
        store = SessionStore()
        assert store.get_turns("never-seen") == []
        assert store.get_last_order_id("never-seen") is None
        assert store.get_last_topic("never-seen") is None

    def test_clear_does_not_affect_other_sessions(self) -> None:
        """Clearing one session must not affect others."""
        store = SessionStore()
        store.add_turn("sess-A", _make_turn(order_id="ORD-1001", topic="returns"))
        store.add_turn("sess-B", _make_turn(order_id="ORD-1002", topic="shipping"))

        store.clear("sess-A")

        assert store.get_turns("sess-A") == []
        assert store.get_last_order_id("sess-A") is None

        # sess-B must be intact
        assert len(store.get_turns("sess-B")) == 1
        assert store.get_last_order_id("sess-B") == "ORD-1002"

    def test_topic_isolation(self) -> None:
        """Topics set in session A must not leak into session B."""
        store = SessionStore()
        store.add_turn("sess-A", _make_turn(topic="returns"))
        store.add_turn("sess-B", _make_turn(topic="shipping"))

        assert store.get_last_topic("sess-A") == "returns"
        assert store.get_last_topic("sess-B") == "shipping"

    def test_two_sessions_no_cross_contamination_direct(self) -> None:
        """Two session_ids in the same store must not share order state.

        This is the unit-level verification of the isolation guarantee
        specified in app/session/store.py.
        """
        store = SessionStore()
        store.add_turn(
            "s-isolation-1",
            _make_turn(order_id="ORD-1001"),
        )
        store.add_turn(
            "s-isolation-2",
            _make_turn(order_id="ORD-1002"),
        )

        assert store.get_last_order_id("s-isolation-1") == "ORD-1001"
        assert store.get_last_order_id("s-isolation-2") == "ORD-1002"
        # No cross-contamination
        assert "ORD-1002" != store.get_last_order_id("s-isolation-1")
        assert "ORD-1001" != store.get_last_order_id("s-isolation-2")

    def test_multiple_turns_per_session_maintains_isolation(self) -> None:
        """Multiple turns in session A must not bleed into session B."""
        store = SessionStore()
        # Session A: 3 turns, each with different order IDs
        for i, oid in enumerate(["ORD-1001", "ORD-1002", "ORD-1003"]):
            store.add_turn("sess-multi-A", _make_turn(order_id=oid))

        # Session B: only 1 turn
        store.add_turn("sess-multi-B", _make_turn(order_id="ORD-1009"))

        # A has 3 turns; B has 1
        assert len(store.get_turns("sess-multi-A")) == 3
        assert len(store.get_turns("sess-multi-B")) == 1

        # Last order in A is ORD-1003 (the most recent)
        assert store.get_last_order_id("sess-multi-A") == "ORD-1003"
        assert store.get_last_order_id("sess-multi-B") == "ORD-1009"
