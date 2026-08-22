"""
Whitelist projection: builds a customer-safe SafeOrderResult from a raw
order dict and a lookup status tag.

Design notes
------------
Projection strategy: EXPLICIT WHITELIST.
  Only the fields enumerated in _copy_top_level() and _build_items() may
  ever appear in the output object.  Any field present in the raw record
  that is NOT listed here is silently ignored — so future additions to
  orders.json are safe by default, not leaked by default.

  We deliberately do NOT read the ``internal`` sub-dict at all — not
  even to discard it.  Treating its content as unreachable means an AI
  injection inside ``warehouse_note`` can never influence the output, even
  if a future refactor accidentally reads from ``internal``.

Cancelled / returned rule:
  If ``status`` is "cancelled" or "returned", ``estimated_delivery`` is
  forced to None and ``customer_safe_message`` is overridden with an
  explicit cancellation/return message, regardless of what the raw record
  contains.  This prevents stale ETA fields from reaching the customer.
"""

from __future__ import annotations

from typing import Literal

from app.schemas import OrderItem, SafeOrderResult

LookupStatus = Literal["missing_id", "invalid_id", "not_found", "found"]

# Statuses that must never carry a delivery estimate.
_TERMINAL_STATUSES = {"cancelled", "returned"}


def _build_items(raw_items: list[dict] | None) -> list[OrderItem]:
    """Convert raw items list to OrderItem objects using the whitelist fields.

    Only ``name``, ``quantity``, and ``final_sale`` are copied.
    ``sku`` (present in raw data) is intentionally excluded per the
    customer-safe whitelist in data/orders-data-dictionary.md.
    """
    if not raw_items:
        return []
    result: list[OrderItem] = []
    for item in raw_items:
        result.append(
            OrderItem(
                name=item["name"],
                quantity=item["quantity"],
                final_sale=item["final_sale"],
            )
        )
    return result


def build_safe_result(
    raw_order: dict | None,
    lookup_status: LookupStatus,
) -> SafeOrderResult:
    """Build a customer-safe SafeOrderResult from a raw record and a status tag.

    Parameters
    ----------
    raw_order:
        The raw dict from orders.json (may contain PII and internal fields),
        or ``None`` when no record was found / input was invalid.
    lookup_status:
        One of ``"missing_id"``, ``"invalid_id"``, ``"not_found"``, ``"found"``.

    Returns
    -------
    SafeOrderResult
        Populated with *only* the whitelisted fields.  All internal data
        (``customer``, ``internal.*``) is unreachable by construction.
    """
    # ------------------------------------------------------------------ #
    # Non-found paths: return a minimal sentinel with a clear message.    #
    # ------------------------------------------------------------------ #
    if lookup_status == "missing_id":
        return SafeOrderResult(
            order_id="",
            status="unknown",
            found=False,
            message="No order ID was provided. Please share your order ID so we can look it up.",
        )

    if lookup_status == "invalid_id":
        return SafeOrderResult(
            order_id="",
            status="unknown",
            found=False,
            message=(
                "The order ID you entered doesn't look right. "
                "Order IDs follow the format ORD-XXXX (e.g. ORD-1001). "
                "Please double-check and try again."
            ),
        )

    if lookup_status == "not_found":
        return SafeOrderResult(
            order_id="",
            status="unknown",
            found=False,
            message=(
                "We couldn't find an order with that ID. "
                "Please verify the order ID and try again."
            ),
        )

    # ------------------------------------------------------------------ #
    # "found" path — strict whitelist projection.                        #
    # ONLY the fields listed below may be copied.                        #
    # ------------------------------------------------------------------ #
    assert raw_order is not None, "raw_order must not be None when lookup_status='found'"

    # --- Whitelisted top-level scalar fields ----------------------------
    order_id: str = raw_order["order_id"]
    status: str = raw_order["status"]
    membership_tier: str | None = raw_order.get("membership_tier")
    placed_at: str | None = raw_order.get("placed_at")
    status_updated_at: str | None = raw_order.get("status_updated_at")
    shipped_at: str | None = raw_order.get("shipped_at")
    delivered_at: str | None = raw_order.get("delivered_at")
    carrier: str | None = raw_order.get("carrier")
    tracking_number: str | None = raw_order.get("tracking_number")
    customer_safe_message: str | None = raw_order.get("customer_safe_message")

    # --- Whitelisted items list -----------------------------------------
    items: list[OrderItem] = _build_items(raw_order.get("items"))

    # --- estimated_delivery: whitelist read, then business-rule override -
    estimated_delivery: str | None = raw_order.get("estimated_delivery")

    if status in _TERMINAL_STATUSES:
        # Force stale ETA to None; override message regardless of raw content.
        estimated_delivery = None
        if status == "cancelled":
            customer_safe_message = "This order was cancelled and will not ship."
        else:  # "returned"
            customer_safe_message = "This order has been returned and will not be re-shipped."

    # --- Assemble output (no field outside this list is touched) --------
    return SafeOrderResult(
        order_id=order_id,
        membership_tier=membership_tier,
        items=items,
        placed_at=placed_at,
        status=status,
        status_updated_at=status_updated_at,
        shipped_at=shipped_at,
        delivered_at=delivered_at,
        carrier=carrier,
        tracking_number=tracking_number,
        estimated_delivery=estimated_delivery,
        customer_safe_message=customer_safe_message,
        found=True,
        message=None,
    )
