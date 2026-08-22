"""
app/session/store.py — In-memory session store for conversation history.

Design
------
Sessions are keyed by an opaque session_id string.  Each session holds an
ordered list of Turn objects, capped at MAX_TURNS (6) to keep prompt
context manageable.

Isolation guarantee: each session_id maps to an independent list; no
shared mutable state exists between sessions (verified by the unit test
``test_session_isolation`` in tests/integration/).

Thread-safety: this implementation is NOT thread-safe.  For a production
deployment behind a multi-threaded or async web server, replace the plain
dict with a thread-local or properly locked store.  For the current
assignment (single-process evaluation), this is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

# Maximum number of turns to retain per session.
MAX_TURNS: int = 6


@dataclass
class Turn:
    """A single completed conversation turn.

    Attributes
    ----------
    user_msg:
        The user's raw message text for this turn.
    agent_response:
        The agent's answer text returned to the user.
    order_id:
        The order ID established during this turn (if any).  Persisted so
        follow-up questions like "When will it arrive?" can reuse it.
    topic:
        A coarse topic label (e.g. "shipping", "returns") established during
        this turn.  Used for follow-up context narrowing.
    """

    user_msg: str
    agent_response: str
    order_id: str | None = None
    topic: str | None = None


class SessionStore:
    """Thread-unsafe in-memory session store keyed by session_id.

    Each session stores at most MAX_TURNS turns; older turns are evicted
    when the limit is exceeded (FIFO).
    """

    def __init__(self) -> None:
        self._store: dict[str, list[Turn]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_turn(self, session_id: str, turn: Turn) -> None:
        """Append *turn* to *session_id*'s history, evicting oldest if needed."""
        if session_id not in self._store:
            self._store[session_id] = []
        turns = self._store[session_id]
        turns.append(turn)
        # Trim to the most recent MAX_TURNS entries.
        if len(turns) > MAX_TURNS:
            self._store[session_id] = turns[-MAX_TURNS:]

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_turns(self, session_id: str) -> list[Turn]:
        """Return a copy of the turn list for *session_id* (oldest first).

        Returns an empty list for an unknown session_id.
        The returned list is a shallow copy — mutating it does not affect
        the store.
        """
        return list(self._store.get(session_id, []))

    def get_last_order_id(self, session_id: str) -> str | None:
        """Return the most recently established order_id for *session_id*.

        Scans turns in reverse order (newest first) and returns the first
        non-None order_id found, or None if no order has been established.
        """
        for turn in reversed(self._store.get(session_id, [])):
            if turn.order_id is not None:
                return turn.order_id
        return None

    def get_last_topic(self, session_id: str) -> str | None:
        """Return the most recently established topic for *session_id*.

        Scans turns in reverse order and returns the first non-None topic.
        """
        for turn in reversed(self._store.get(session_id, [])):
            if turn.topic is not None:
                return turn.topic
        return None

    def build_history(self, session_id: str) -> list[dict[str, str]]:
        """Build a list of {role, content} dicts for the LLM call.

        The returned list alternates user / model roles.  The current turn's
        user message is NOT included here — it is added by the orchestrator
        just before calling the LLM.

        Returns
        -------
        list[dict[str, str]]
            Prior turns formatted as ``[{"role": "user", "content": "..."},
            {"role": "model", "content": "..."}, ...]``.
        """
        history: list[dict[str, str]] = []
        for turn in self._store.get(session_id, []):
            history.append({"role": "user", "content": turn.user_msg})
            history.append({"role": "model", "content": turn.agent_response})
        return history

    def clear(self, session_id: str) -> None:
        """Remove all turns for *session_id* (useful for testing)."""
        self._store.pop(session_id, None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global session store.  Import this from other modules rather than
#: instantiating a new SessionStore per call.
SESSION_STORE: SessionStore = SessionStore()
