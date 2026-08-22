"""
Order ID normalisation and validation.

ID format confirmed from data/orders.json inspection:
  "ORD-" followed by exactly 4 decimal digits, e.g. "ORD-1001".

Design notes
------------
* normalize_order_id() performs the minimum mutations required to accept
  plausible user input (trim whitespace, uppercase) and then validates
  the result against the confirmed pattern.
* On success it returns a 2-tuple ("ok", normalised_id).
* On malformed input it returns a 2-tuple ("invalid", original_input)
  — never raises, never silently continues.
* "Empty or missing" input is detected before pattern validation and
  returns ("missing_id", ""), keeping the three error signals distinct
  for the orchestrator layer.
"""

from __future__ import annotations

import re
from typing import Literal

# Exact pattern confirmed by inspecting every order_id in data/orders.json:
# uppercase "ORD-" followed by exactly 4 decimal digits.
_ORDER_ID_RE = re.compile(r"^ORD-\d{4}$")

NormaliseStatus = Literal["ok", "invalid", "missing_id"]


def normalize_order_id(raw: str | None) -> tuple[NormaliseStatus, str]:
    """Normalise and validate a raw order-ID string supplied by a caller.

    Returns a ``(status, value)`` tuple:

    * ``("ok", normalised_id)``       — valid, use *normalised_id* for lookup.
    * ``("invalid", raw_input)``      — does not match the expected pattern.
    * ``("missing_id", "")``          — blank / None input; caller should ask
                                        the user to supply an order ID.

    The function never raises; error cases are surfaced through the status tag
    so that the orchestrator can phrase a precise response without an
    ``except`` block.
    """
    # --- 1. Detect missing/empty input ------------------------------------
    if not raw or not raw.strip():
        return ("missing_id", "")

    # --- 2. Normalise: strip surrounding whitespace, uppercase ------------
    normalised = raw.strip().upper()

    # --- 3. Validate against the confirmed pattern -----------------------
    if _ORDER_ID_RE.match(normalised):
        return ("ok", normalised)

    return ("invalid", raw)
