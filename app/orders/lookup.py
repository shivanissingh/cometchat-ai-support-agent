"""
Order data loading and lookup.

Loading strategy
----------------
The orders dataset (data/orders.json) is loaded **once at module import
time** and stored in a module-level dict keyed by order_id.  This avoids
repeated disk I/O on every call and keeps latency low for the orchestrator.
A simple in-process cache is sufficient for the current dataset size; if
hot-reload is ever needed, replace the module-level constant with an
``@functools.lru_cache``-decorated loader or a startup hook.

Public API
----------
lookup_order_raw(order_id) -> Optional[dict]
    Low-level: returns the raw dict (including internal fields) or None.
    Only projection.py is authorised to consume the output.

lookup_order(order_id_raw) -> SafeOrderResult
    High-level entrypoint for the orchestrator.  Composes:
      normalize_order_id → lookup_order_raw → build_safe_result
    This is the function agents should call; it guarantees that the
    result is already projected and safe.

Three-way error signals
-----------------------
The orchestrator needs to phrase distinct responses for:
  1. "missing_id"  — caller passed blank/None → "please give me your order ID"
  2. "invalid_id"  — input doesn't match ORD-XXXX → "please double-check it"
  3. "not_found"   — well-formed ID but not in the dataset → "not found"
These are conveyed through SafeOrderResult.found=False and .message, and
also through the lookup_status string passed to build_safe_result.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.orders.normalize import normalize_order_id
from app.orders.projection import build_safe_result
from app.schemas import SafeOrderResult

# ---------------------------------------------------------------------------
# One-time load at module import
# ---------------------------------------------------------------------------
_DATA_FILE = Path(__file__).parents[2] / "data" / "orders.json"


def _load_orders() -> dict[str, dict]:
    """Load orders.json once and index by order_id."""
    with _DATA_FILE.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return {order["order_id"]: order for order in payload["orders"]}


# Module-level constant: populated once, read-only thereafter.
_ORDERS: dict[str, dict] = _load_orders()


# ---------------------------------------------------------------------------
# Low-level raw lookup
# ---------------------------------------------------------------------------


def lookup_order_raw(order_id: str) -> dict | None:
    """Return the raw order dict for *order_id*, or ``None`` if not found.

    Parameters
    ----------
    order_id:
        A *normalised* order ID (output of normalize_order_id).

    Returns
    -------
    dict | None
        The raw record from orders.json — may contain PII and internal
        fields.  Only ``projection.py`` is authorised to consume this.

    Notes
    -----
    Missing / None input is deliberately not handled here; callers must
    call ``normalize_order_id`` first.  The three-way error distinction
    (missing_id / invalid_id / not_found) is managed at the
    ``lookup_order`` level.
    """
    return _ORDERS.get(order_id)


# ---------------------------------------------------------------------------
# High-level public entrypoint (orchestrator calls this)
# ---------------------------------------------------------------------------


def lookup_order(order_id_raw: str | None) -> SafeOrderResult:
    """Compose normalise → raw-lookup → safe-projection into one call.

    This is the single function the orchestrator agent should call.
    It guarantees:
      * The returned ``SafeOrderResult`` contains no PII or internal fields.
      * The three error cases (missing_id, invalid_id, not_found) produce
        distinct ``SafeOrderResult`` values with appropriate messages.

    Parameters
    ----------
    order_id_raw:
        The order ID exactly as received from the user (may be ``None``,
        blank, malformed, or correctly formatted).

    Returns
    -------
    SafeOrderResult
        Always returns a valid model instance; never raises on bad input.
    """
    # Step 1: normalise & validate
    norm_status, normalised = normalize_order_id(order_id_raw)  # type: ignore[arg-type]

    if norm_status == "missing_id":
        return build_safe_result(None, "missing_id")

    if norm_status == "invalid":
        return build_safe_result(None, "invalid_id")

    # Step 2: raw lookup (normalised ID is guaranteed well-formed here)
    raw = lookup_order_raw(normalised)

    if raw is None:
        return build_safe_result(None, "not_found")

    # Step 3: project to safe output
    return build_safe_result(raw, "found")
