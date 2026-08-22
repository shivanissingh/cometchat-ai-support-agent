"""
app/agent/router.py — Deterministic pre-router.

All routing decisions are made BEFORE any LLM call.  The router inspects
the current user message and recent session context with regex/keyword rules
and returns a RouterDecision dataclass.  Every branch emits a structured
trace event at stage="router" so evaluation and debugging can verify routing
decisions without inspecting LLM output.

Routing priority (highest to lowest)
-------------------------------------
a. ORDER_DIRECT   — current message contains a valid ORD-XXXX token
b. ORDER_ASK_ID  — current message contains order-intent keywords but no
                    order ID is present in this turn or recent session context
c. ORDER_FOLLOWUP — session context established an order ID in a prior turn
                    (e.g. "When will it arrive?" after "Where is ORD-1007?")
d. KNOWLEDGE      — none of the above; treat as a knowledge-path query

Design note on the fallback function-calling path (KNOWLEDGE branch)
----------------------------------------------------------------------
After the LLM generates a knowledge-path response, the orchestrator checks
whether the model emitted a ``lookup_order`` function call.  If so, it
executes the order tool and re-calls the LLM with the result appended.
This fallback should rarely fire — it exists only as a safety net for
edge cases the regex missed.

Security note
-------------
The regex extracts an order ID token from the user message for the
ORDER_DIRECT and ORDER_FOLLOWUP paths.  The extracted value is always
validated by ``normalize_order_id`` before being passed to the order tool
— the router never directly passes a raw user-supplied string to the tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from app.observability import emit_trace
from app.orders.normalize import normalize_order_id
from app.schemas import TraceEvent

# ---------------------------------------------------------------------------
# Order-ID pattern (must match normalize.py exactly: ORD-\d{4})
# ---------------------------------------------------------------------------

_ORDER_ID_RE = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Order-intent keywords (case-insensitive)
# ---------------------------------------------------------------------------

_ORDER_INTENT_RE = re.compile(
    r"\b(order|track|tracking|shipment|shipments|delivery|deliveries|"
    r"ship(?:ped|ping)|where\s+is|package|packages|status|dispatch(?:ed)?|"
    r"arriv(?:e|ed|al)|when\s+will|estimated?|eta)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# RouterDecision
# ---------------------------------------------------------------------------

RouterPath = Literal["order_direct", "order_ask_id", "order_followup", "knowledge"]


@dataclass
class RouterDecision:
    """The result of the deterministic pre-routing pass.

    Attributes
    ----------
    path:
        Which routing branch was selected.
    order_id:
        The normalised order ID to look up (only set for order_direct and
        order_followup paths).
    reason:
        A short human-readable explanation of why this path was chosen
        (logged in the trace event payload).
    """

    path: RouterPath
    order_id: str | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route(
    session_id: str,
    turn_id: int,
    user_message: str,
    last_order_id: str | None,
) -> RouterDecision:
    """Determine the routing path for the current turn deterministically.

    Parameters
    ----------
    session_id:
        Current session identifier (used for trace events).
    turn_id:
        Current turn number within the session (0-based).
    user_message:
        The raw user message text for this turn.
    last_order_id:
        The most recently established order ID from session history, or None.

    Returns
    -------
    RouterDecision
        The selected path with optional order_id and a reason string.
    """
    decision = _make_decision(user_message, last_order_id)

    emit_trace(
        TraceEvent(
            session_id=session_id,
            turn_id=turn_id,
            stage="router",
            payload={
                "path": decision.path,
                "order_id": decision.order_id,
                "reason": decision.reason,
                "message_snippet": user_message[:120],
            },
            timestamp=datetime.now(UTC).isoformat(),
        )
    )

    return decision


def _make_decision(user_message: str, last_order_id: str | None) -> RouterDecision:
    """Pure decision logic (no side effects) — separated for testability."""

    # ------------------------------------------------------------------
    # Branch (a): ORDER_DIRECT
    # The current message contains a token matching ORD-\d{4}.
    # Validate via normalize_order_id so we know the token is well-formed.
    # ------------------------------------------------------------------
    match = _ORDER_ID_RE.search(user_message)
    if match:
        raw_id = match.group(0)
        status, normalised = normalize_order_id(raw_id)
        if status == "ok":
            return RouterDecision(
                path="order_direct",
                order_id=normalised,
                reason=f"Order ID {normalised!r} found in current message (regex match).",
            )

    # ------------------------------------------------------------------
    # Branch (b): ORDER_ASK_ID
    # Order-intent keywords present, but no order ID anywhere in this turn
    # or in recent session history.
    # ------------------------------------------------------------------
    has_intent = bool(_ORDER_INTENT_RE.search(user_message))
    if has_intent and last_order_id is None:
        return RouterDecision(
            path="order_ask_id",
            order_id=None,
            reason=(
                "Order-intent keywords found; no order ID in current message"
                " or session history."
            ),
        )

    # ------------------------------------------------------------------
    # Branch (c): ORDER_FOLLOWUP
    # No ID in the current message, but the session established one earlier.
    # Two sub-conditions trigger this:
    #   (c1) order-intent keywords present in the current message, OR
    #   (c2) the message is a short conversational follow-up (<=12 words)
    #        e.g. "When will it arrive?", "Any updates?" that naturally
    #        continues an established order context.
    # ------------------------------------------------------------------
    if last_order_id is not None:
        word_count = len(user_message.split())
        is_short_followup = word_count <= 12
        if has_intent or is_short_followup:
            trigger = (
                "order-intent keywords"
                if has_intent
                else f"short follow-up message ({word_count} words)"
            )
            return RouterDecision(
                path="order_followup",
                order_id=last_order_id,
                reason=(
                    f"Reusing session order ID {last_order_id!r} from prior"
                    f" turn ({trigger})."
                ),
            )

    # ------------------------------------------------------------------
    # Branch (d): KNOWLEDGE
    # Default — treat as a knowledge-path query.
    # ------------------------------------------------------------------
    return RouterDecision(
        path="knowledge",
        order_id=None,
        reason="No order ID or order-intent keywords found; routing to knowledge path.",
    )
