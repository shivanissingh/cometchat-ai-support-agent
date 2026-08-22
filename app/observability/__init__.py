"""
app/observability/__init__.py — Structured trace event emission.

All pipeline stages (router, retrieval, precedence, conflict, tool_call,
tool_result, llm_call, validation, response) call emit_trace() to produce
a machine-readable audit trail in the structured JSON log stream.

Security rule: never include GEMINI_API_KEY or forbidden order fields
(customer_email, address, internal_notes, risk_score) in any payload.
"""

from __future__ import annotations

import logging

from app.schemas import TraceEvent

_logger = logging.getLogger("observability")


def emit_trace(event: TraceEvent) -> None:
    """Emit a TraceEvent as a structured JSON log line.

    Parameters
    ----------
    event:
        A fully-populated TraceEvent instance.  The caller is responsible
        for never including sensitive data (API keys, PII, forbidden order
        fields) in ``event.payload``.
    """
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
