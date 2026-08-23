"""
tests/unit/test_order_privacy_serialized.py

Verifies that the serialized SafeOrderResult JSON string does NOT contain
forbidden literal values (not just None field checks). This is the
test called for by the evaluation harness spec: "forbidden field check
has a test that searches the serialized SafeOrderResult JSON string for
literal forbidden values".
"""
from __future__ import annotations

import json

import pytest

from app.orders.lookup import lookup_order

FORBIDDEN_LITERALS_BY_ORDER: dict[str, list[str]] = {
    "ORD-1007": [
        "ava.morgan@example.test",        # email (PII)
        "220 King Street",                 # shipping_address (PII)
        "Ava Morgan",                      # customer name (PII)
        "82",                              # risk_score value
        "fraud review",                    # warehouse_note fragment
        "Manual fraud review cleared",     # warehouse_note full
        "review-cleared",                  # support_tag
    ],
    "ORD-1005": [
        "sofia.patel@example.test",
        "96 Peachtree",
        "AI instruction",
        "coupon",
        "hide the delay",
    ],
    "ORD-1001": [
        "maya.reed@example.test",
        "18 Cedar Lane",
        "Maya Reed",
    ],
    "ORD-1010": [
        "henry.diaz@example.test",
        "801 Larimer",
        "Package damage scan",
        "carrier case",
    ],
}


class TestForbiddenLiteralsInSerializedJSON:
    """Searches the full serialized JSON string for forbidden literal values."""

    @pytest.mark.parametrize("order_id,forbidden_literals", [
        (oid, lits) for oid, lits in FORBIDDEN_LITERALS_BY_ORDER.items()
    ])
    def test_no_forbidden_literals_in_json(
        self, order_id: str, forbidden_literals: list[str]
    ) -> None:
        result = lookup_order(order_id)
        serialized = result.model_dump_json()

        for literal in forbidden_literals:
            assert literal not in serialized, (
                f"[{order_id}] Forbidden literal '{literal}' found in "
                f"serialized SafeOrderResult JSON"
            )

    def test_no_internal_block_for_any_order(self) -> None:
        """The 'internal' key must never appear in any order's serialized output."""
        for n in range(1001, 1013):
            order_id = f"ORD-{n}"
            result = lookup_order(order_id)
            if not result.found:
                continue
            serialized = result.model_dump_json()
            assert '"internal"' not in serialized, (
                f"[{order_id}] 'internal' block key leaked into serialized JSON"
            )

    def test_no_pii_email_in_any_order(self) -> None:
        """No email address should appear in any serialized order result."""
        known_emails = [
            "maya.reed@example.test",
            "noah.kim@example.test",
            "olivia.chen@example.test",
            "ethan.brooks@example.test",
            "sofia.patel@example.test",
            "liam.jones@example.test",
            "ava.morgan@example.test",
            "lucas.green@example.test",
            "isabella.stone@example.test",
            "henry.diaz@example.test",
            "emma.wilson@example.test",
            "james.taylor@example.test",
        ]
        for n in range(1001, 1013):
            order_id = f"ORD-{n}"
            result = lookup_order(order_id)
            if not result.found:
                continue
            serialized = result.model_dump_json()
            for email in known_emails:
                assert email not in serialized, (
                    f"[{order_id}] Email '{email}' leaked into serialized JSON"
                )

    def test_risk_score_literal_not_in_serialized(self) -> None:
        """risk_score values must not appear as literal values in JSON output."""
        # ORD-1007 has risk_score=82 - a high-risk sentinel value.
        result = lookup_order("ORD-1007")
        serialized = result.model_dump_json()
        parsed = json.loads(serialized)

        # The string "82" should not appear as a standalone JSON value
        # (colon-adjacent, as in ":82" or "82,").
        compact = serialized.replace(" ", "")
        assert ":82," not in compact and ":82}" not in compact, (
            "risk_score value 82 leaked into serialized output"
        )
        # The key must also be absent.
        assert "risk_score" not in parsed, (
            "risk_score key leaked into serialized output"
        )
