"""
app/observability/__init__.py — Structured trace event emission.

All pipeline stages (router, retrieval, precedence, conflict, tool_call,
tool_result, llm_call, validation, response) call emit_trace() to produce
a machine-readable audit trail in the structured JSON log stream.

Security rule: never include GEMINI_API_KEY or forbidden order fields
(customer_email, address, internal_notes, risk_score) in any payload.
Scrubbing is applied automatically via ``tracer.scrub_payload`` so callers
do not need to pre-scrub their payloads.
"""

from __future__ import annotations

import logging

from app.schemas import TraceEvent

_logger = logging.getLogger("observability")


def emit_trace(event: TraceEvent) -> None:
    """Emit a TraceEvent as a scrubbed structured JSON log line.

    The payload is passed through ``scrub_payload`` (from
    ``app.observability.tracer``) before being written to the log so that
    forbidden order fields and API key values are never recorded.

    Parameters
    ----------
    event:
        A fully-populated TraceEvent instance.
    """
    # Import here to avoid a circular import (tracer imports schemas which
    # imports nothing from observability, but belt-and-suspenders guard).
    from app.observability.tracer import scrub_payload

    safe_payload = scrub_payload(event.payload)
    _logger.info(
        "trace",
        extra={
            "session_id": event.session_id,
            "turn_id": event.turn_id,
            "stage": event.stage,
            "payload": safe_payload,
            "event_timestamp": event.timestamp,
        },
    )
