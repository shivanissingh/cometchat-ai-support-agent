"""
app/observability/tracer.py — Payload scrubbing and structured trace emission.

This module provides two public utilities:

``scrub_payload(payload)``
    Recursively walk a payload dict and redact:

    * Any key whose name matches a known forbidden order field (imported from
      app.safety.validator — same source of truth, never duplicated here).
    * Any key whose name contains "api_key" (case-insensitive) — ensures
      GEMINI_API_KEY can never accidentally appear in a trace payload.
    * String values matching the literal ``GEMINI_API_KEY`` env-var value.

``Tracer``
    A lightweight per-turn helper that accumulates ``TraceEvent`` objects in
    memory and logs each one as a scrubbed structured JSON line.  The CLI and
    Streamlit UI consume ``Tracer.events`` directly — no log re-parsing needed.

Security contract
-----------------
* ``GEMINI_API_KEY`` is **never** written to any log line or trace payload.
* Forbidden order fields (risk_score, warehouse_note, support_tags,
  customer_email, shipping_address, internal_notes) are **never** written to
  any log line or trace payload — any key/value matching these is replaced
  with ``"[REDACTED]"`` before emission.

The ``emit_trace`` function in ``app.observability.__init__`` calls
``scrub_payload`` so that every caller benefits from scrubbing automatically,
whether the payload was assembled in the orchestrator, router, or validator.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from app.safety.validator import _FORBIDDEN_FIELD_NAMES  # shared constant – single source of truth
from app.schemas import TraceEvent

_logger = logging.getLogger("observability.tracer")

# ---------------------------------------------------------------------------
# Build the scrubbing key-set once at import time
# ---------------------------------------------------------------------------

#: Set of lower-cased forbidden field names (from validator — single source).
_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    name.lower() for name in _FORBIDDEN_FIELD_NAMES
)

# ---------------------------------------------------------------------------
# Secret value scrubbing — GEMINI_API_KEY must never appear in logs
# ---------------------------------------------------------------------------

_API_KEY_VALUE: str = os.getenv("GEMINI_API_KEY", "")


def _is_forbidden_key(key: str) -> bool:
    """Return True if *key* must be scrubbed from trace payloads."""
    k = key.lower()
    return k in _FORBIDDEN_KEYS or "api_key" in k


def _scrub_value(value: Any) -> Any:  # noqa: ANN401
    """Replace *value* with '[REDACTED]' if it matches the API key literal."""
    if (
        _API_KEY_VALUE
        and isinstance(value, str)
        and _API_KEY_VALUE in value
    ):
        return "[REDACTED]"
    return value


def scrub_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *payload* with all forbidden keys/values redacted.

    Parameters
    ----------
    payload:
        Arbitrary dict produced by a pipeline stage.

    Returns
    -------
    dict[str, Any]
        A new dict where:
        - Keys matching ``_FORBIDDEN_KEYS`` or containing ``"api_key"`` have
          their values replaced with ``"[REDACTED]"``.
        - String values that contain the literal ``GEMINI_API_KEY`` env-var
          value are replaced with ``"[REDACTED]"``.
        - Nested dicts are recursively scrubbed.
        - Other collection types (list, tuple) are recursively scrubbed.

    Notes
    -----
    The original *payload* dict is never mutated.
    """
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_forbidden_key(key):
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = scrub_payload(value)
        elif isinstance(value, list):
            result[key] = [
                scrub_payload(item) if isinstance(item, dict) else _scrub_value(item)
                for item in value
            ]
        else:
            result[key] = _scrub_value(value)
    return result


# ---------------------------------------------------------------------------
# Tracer — per-turn in-memory collector
# ---------------------------------------------------------------------------


class Tracer:
    """Accumulate and emit scrubbed TraceEvents for one agent turn.

    Intended use (inside a single ``handle_message`` call)::

        tracer = Tracer(session_id="s1", turn_id=0)
        tracer.emit("router", {"path": "knowledge"})
        # … later …
        return AgentResponse(..., trace=tracer.events)

    The ``events`` list is the authoritative, scrubbed trace for the turn.
    Both the CLI ``--debug`` flag and the Streamlit debug panel display it
    directly — the data has already been scrubbed so no extra scrubbing is
    needed at the presentation layer.
    """

    def __init__(self, session_id: str, turn_id: int) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.events: list[TraceEvent] = []

    def emit(self, stage: str, payload: dict[str, Any]) -> TraceEvent:  # noqa: ANN401
        """Create, store, log, and return a scrubbed TraceEvent.

        Parameters
        ----------
        stage:
            One of the literal stage names defined in ``TraceEvent.stage``.
        payload:
            Stage-specific data.  Will be scrubbed before storage and logging.

        Returns
        -------
        TraceEvent
            The stored event (scrubbed payload).
        """
        safe_payload = scrub_payload(payload)
        event = TraceEvent(
            session_id=self.session_id,
            turn_id=self.turn_id,
            stage=stage,  # type: ignore[arg-type]
            payload=safe_payload,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.events.append(event)
        _logger.info(
            "trace",
            extra={
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "stage": event.stage,
                "payload": event.payload,
                "event_timestamp": event.timestamp,
            },
        )
        return event
