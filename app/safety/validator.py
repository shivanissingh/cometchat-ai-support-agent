"""
app/safety/validator.py — Post-processing safety validator.

This module is a safety NET, not the primary defence.  The primary defences are:

1. SafeOrderResult (projection.py) — PII and internal fields are structurally
   absent from the DTO; they can never reach the prompt.
2. Evidence pack labelling (prompts.py) — retrieved text is clearly marked as
   REFERENCE DATA, not instructions.

This validator adds a final, belt-and-suspenders check that catches any leakage
that might occur through unexpected code paths or future refactors.

Three enforcement layers
------------------------
a. Forbidden field scan — redacts any mention of forbidden field names or
   known sensitive patterns in the LLM output.
b. Citation enforcement — if authoritative sources were used but the LLM
   output contains no [filename — heading] citation, one is appended
   deterministically from the evidence pack.
c. Action claim enforcement — replaces any claim of a completed action
   (cancel, refund, etc.) with a fixed handoff message, since no action tool
   exists in this system.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from app.observability import emit_trace
from app.schemas import RetrievedChunk, TraceEvent

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Forbidden field names — these must never appear verbatim in LLM output.
_FORBIDDEN_FIELD_NAMES: list[str] = [
    "customer_email",
    "shipping_address",
    "internal_notes",
    "risk_score",
    "warehouse_note",
    "support_tags",
    "internal_notes",
]

# Build a compiled pattern for forbidden field names.
_FORBIDDEN_FIELDS_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _FORBIDDEN_FIELD_NAMES) + r")\b",
    re.IGNORECASE,
)

# Citation pattern: [filename — heading] (em dash or double hyphen).
_CITATION_RE = re.compile(r"\[.+?[—\-]{1,2}.+?\]")

# Phrases indicating a claimed completed action.
_ACTION_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"i(?:'ve| have)\s+cancel(?:led|ed)",
        r"i(?:'ve| have)\s+issued\s+(?:your|a|the)\s+refund",
        r"i(?:'ve| have)\s+processed\s+(?:your|a|the)\s+(?:refund|return|cancellation)",
        # Broad catch-all: "I've processed a <anything>" e.g. coupon, discount
        r"i(?:'ve| have)\s+processed\s+(?:a|an|your)\s+\w+",
        r"i(?:'ve| have)\s+placed\s+(?:your|a|the)\s+(?:order|replacement)",
        r"i(?:'ve| have)\s+applied\s+(?:a|an|your|the)\s+(?:coupon|discount|credit|promo)",
        r"i(?:'ve| have)\s+added\s+(?:a|an|your|the)\s+(?:coupon|discount|credit|promo)",
        r"cancellation\s+has\s+been\s+(?:processed|completed|confirmed)",
        r"refund\s+has\s+been\s+(?:issued|processed|approved)",
        r"your\s+order\s+has\s+been\s+cancel(?:led|ed)\s+by\s+(?:me|us|our\s+system)",
        r"i(?:'ve| have)\s+(?:submitted|filed|raised)"
        r"\s+(?:your|a|the)\s+(?:claim|ticket|request)",
        r"i(?:'ve| have)\s+(?:updated|changed|modified)"
        r"\s+(?:your|the)\s+(?:address|shipping|order)",
    ]
]

_ACTION_HANDOFF_MSG = (
    "I'm not able to perform that action directly through this chat. "
    "Please contact our support team who can assist you with this request."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_response(
    answer: str,
    auth_chunks: list[RetrievedChunk],
    session_id: str,
    turn_id: int,
) -> str:
    """Post-process *answer* through all three safety layers.

    Parameters
    ----------
    answer:
        The raw text response from the LLM (or a deterministic response).
    auth_chunks:
        Authoritative chunks that were used in this turn (may be empty for
        order-path or abstention responses).
    session_id:
        Current session ID (for trace events).
    turn_id:
        Current turn number (for trace events).

    Returns
    -------
    str
        The validated (and possibly modified) response text.
    """
    original = answer
    warnings: list[str] = []

    # --- Layer (a): Forbidden field scan ---------------------------------
    answer, field_warnings = _redact_forbidden_fields(answer)
    warnings.extend(field_warnings)

    # --- Layer (b): Citation enforcement ---------------------------------
    answer, cite_warnings = _enforce_citations(answer, auth_chunks)
    warnings.extend(cite_warnings)

    # --- Layer (c): Action claim enforcement -----------------------------
    answer, action_warnings = _enforce_no_action_claims(answer)
    warnings.extend(action_warnings)

    # Emit a trace event if any modification was made.
    if warnings:
        emit_trace(
            TraceEvent(
                session_id=session_id,
                turn_id=turn_id,
                stage="validation",
                payload={
                    "warnings": warnings,
                    "original_length": len(original),
                    "final_length": len(answer),
                },
                timestamp=datetime.now(UTC).isoformat(),
            )
        )
        for w in warnings:
            _logger.warning("Validation warning", extra={"warning": w, "session_id": session_id})

    return answer


# ---------------------------------------------------------------------------
# Layer implementations
# ---------------------------------------------------------------------------


def _redact_forbidden_fields(text: str) -> tuple[str, list[str]]:
    """Redact any forbidden field names from *text*.

    Returns the (possibly modified) text and a list of warning strings.
    """
    warnings: list[str] = []

    def _replacer(m: re.Match[str]) -> str:
        warnings.append(f"Forbidden field name redacted: {m.group(0)!r}")
        return "[REDACTED]"

    result = _FORBIDDEN_FIELDS_RE.sub(_replacer, text)
    return result, warnings


def _enforce_citations(
    text: str,
    auth_chunks: list[RetrievedChunk],
) -> tuple[str, list[str]]:
    """Append a citation if auth chunks were used but the text has none.

    Returns the (possibly modified) text and a list of warning strings.
    """
    warnings: list[str] = []

    if not auth_chunks:
        # No authoritative chunks were used — citation not required.
        return text, warnings

    if _CITATION_RE.search(text):
        # At least one citation present — no action needed.
        return text, warnings

    # No citation found — append one from the top authoritative chunk.
    top = auth_chunks[0]
    citation = f"[{top.chunk.filename} — {top.chunk.heading_path}]"
    warnings.append(f"Citation missing; appended deterministically: {citation}")
    text = text.rstrip() + f"\n\n*Source: {citation}*"
    return text, warnings


def _enforce_no_action_claims(text: str) -> tuple[str, list[str]]:
    """Replace any completed-action claim with the handoff message.

    If ANY action-claim pattern matches, the entire answer is replaced with
    the handoff message — a partial replacement would leave confusing
    sentence fragments.

    Returns the (possibly modified) text and a list of warning strings.
    """
    warnings: list[str] = []

    for pattern in _ACTION_CLAIM_PATTERNS:
        if pattern.search(text):
            warnings.append(
                f"Action claim detected (pattern: {pattern.pattern!r}); "
                "replaced with handoff message."
            )
            return _ACTION_HANDOFF_MSG, warnings

    return text, warnings
