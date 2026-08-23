"""
tests/unit/test_schema_b_harness.py

Unit tests confirming that the evaluation harness correctly asserts Turn 1
and Turn 2 independently when processing Schema B cases.

This covers the harness spec requirement: "per-turn assertion path (Schema B)
has a unit test confirming the harness correctly asserts Turn 1 and Turn 2
independently."
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import TraceEvent
from evaluation_runner.run_eval import (
    _assert_turn,
    _check_must_include,
    _check_must_not_include,
    _check_required_sources,
    _check_tool,
)

# ---------------------------------------------------------------------------
# Minimal stub for AgentResponse
# ---------------------------------------------------------------------------


@dataclass
class _StubResponse:
    """Minimal stand-in for AgentResponse used in harness unit tests."""

    answer: str
    citations: list[dict] = field(default_factory=list)
    handoff: bool = False
    trace: list = field(default_factory=list)


def _empty_stats() -> dict:
    return {"hits": 0, "total": 0}


def _make_tool_event(order_id: str) -> TraceEvent:
    return TraceEvent(
        session_id="test",
        turn_id=0,
        stage="tool_call",
        payload={"function": "lookup_order", "order_id": order_id},
        timestamp="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Tests for individual assertion helpers
# ---------------------------------------------------------------------------


class TestMustInclude:
    def test_phrase_present_passes(self) -> None:
        assert _check_must_include("order was delivered", ["delivered"]) is None

    def test_phrase_missing_fails(self) -> None:
        reason = _check_must_include("order is processing", ["delivered"])
        assert reason is not None
        assert "delivered" in reason

    def test_case_insensitive(self) -> None:
        assert _check_must_include("Order Was Delivered", ["delivered"]) is None

    def test_multiple_phrases_all_must_match(self) -> None:
        reason = _check_must_include("delivered via UPS", ["delivered", "August 22"])
        assert reason is not None
        assert "August 22" in reason

    def test_all_phrases_present_passes(self) -> None:
        assert (
            _check_must_include(
                "delivered via UPS on August 22", ["delivered", "August 22"]
            )
            is None
        )


class TestMustNotInclude:
    def test_phrase_absent_passes(self) -> None:
        assert (
            _check_must_not_include("order is processing", ["your return is approved"])
            is None
        )

    def test_phrase_present_fails(self) -> None:
        reason = _check_must_not_include(
            "your return is approved", ["your return is approved"]
        )
        assert reason is not None

    def test_case_insensitive(self) -> None:
        reason = _check_must_not_include(
            "Your Return Is Approved", ["your return is approved"]
        )
        assert reason is not None


class TestCheckTool:
    def test_order_lookup_found_in_trace(self) -> None:
        trace = [_make_tool_event("ORD-1007")]
        assert _check_tool(trace, "order_lookup", "") is None

    def test_order_lookup_missing_from_trace_fails(self) -> None:
        reason = _check_tool([], "order_lookup", "")
        assert reason is not None

    def test_not_called_with_no_events_passes(self) -> None:
        assert _check_tool([], "not_called", "") is None

    def test_not_called_but_event_present_fails(self) -> None:
        trace = [_make_tool_event("ORD-1007")]
        reason = _check_tool(trace, "not_called", "")
        assert reason is not None

    def test_not_called_without_id_asks_for_order_id(self) -> None:
        assert (
            _check_tool([], "not_called_without_id", "Please provide your order ID")
            is None
        )

    def test_not_called_without_id_no_ask_fails(self) -> None:
        reason = _check_tool([], "not_called_without_id", "I cannot help with that")
        assert reason is not None


class TestRequiredSources:
    def test_source_present_passes(self) -> None:
        citations = [{"filename": "01-returns-policy-current.md", "heading": ""}]
        r, hits, misses = _check_required_sources(
            citations, ["01-returns-policy-current.md"]
        )
        assert r is None
        assert hits == ["01-returns-policy-current.md"]
        assert misses == []

    def test_source_missing_fails(self) -> None:
        r, hits, misses = _check_required_sources([], ["01-returns-policy-current.md"])
        assert r is not None
        assert misses == ["01-returns-policy-current.md"]


# ---------------------------------------------------------------------------
# Tests for Schema B per-turn assertion logic
# ---------------------------------------------------------------------------


class TestSchemaBAssertTurn:
    """Verify _assert_turn works independently for each turn."""

    def test_turn_1_pass_turn_2_fail_catches_turn_2(self) -> None:
        """Schema B: a failing Turn 2 must be reported even if Turn 1 passes."""
        turn_1_expect = {"must_include": ["delivered"]}
        turn_2_expect = {"must_include": ["30 calendar days"]}

        turn_1_resp = _StubResponse(answer="The order was delivered on August 10.")
        turn_2_resp = _StubResponse(answer="You can return items anytime.")

        t1_failures = _assert_turn(
            turn_1_expect, turn_1_resp, [], "model", "",
            _empty_stats(), _empty_stats(),
        )
        assert t1_failures == [], f"Turn 1 should pass, got: {t1_failures}"

        t2_failures = _assert_turn(
            turn_2_expect, turn_2_resp, [], "model", "",
            _empty_stats(), _empty_stats(),
        )
        assert t2_failures, "Turn 2 should fail"
        assert "30 calendar days" in t2_failures[0]

    def test_turn_1_fail_captured_independently(self) -> None:
        """Schema B: a failing Turn 1 must be captured independently of Turn 2."""
        turn_1_expect = {
            "must_include": ["delivered"],
            "must_not_include": ["risk score"],
        }
        turn_1_resp = _StubResponse(answer="Your risk score is low.")

        failures = _assert_turn(
            turn_1_expect, turn_1_resp, [], "model", "",
            _empty_stats(), _empty_stats(),
        )
        assert any("risk score" in f for f in failures), (
            f"Expected risk score failure, got: {failures}"
        )

    def test_both_turns_pass(self) -> None:
        """Schema B: both turns passing means no failures."""
        turn_1_expect = {"must_include": ["delivered"]}
        turn_2_expect = {"must_not_include": ["your return is approved"]}

        turn_1_resp = _StubResponse(answer="Your order was delivered on August 10.")
        turn_2_resp = _StubResponse(
            answer="Returns are subject to our standard 30-day policy."
        )

        t1_failures = _assert_turn(
            turn_1_expect, turn_1_resp, [], "model", "",
            _empty_stats(), _empty_stats(),
        )
        t2_failures = _assert_turn(
            turn_2_expect, turn_2_resp, [], "model", "",
            _empty_stats(), _empty_stats(),
        )
        assert t1_failures == []
        assert t2_failures == []

    def test_tool_assertion_on_turn_1_only(self) -> None:
        """Schema B: tool assertion for Turn 1 should not affect Turn 2."""
        turn_1_expect = {
            "tool": "order_lookup",
            "tool_arguments": {"order_id": "ORD-1006"},
        }
        turn_2_expect = {
            "tool": "not_called",
            "must_not_include": ["your return is approved"],
        }

        turn_1_resp = _StubResponse(
            answer="ORD-1006 was delivered August 10.",
            trace=[_make_tool_event("ORD-1006")],
        )
        turn_2_resp = _StubResponse(
            answer="You can return items within 30 days of delivery.",
            trace=[],
        )

        t1_failures = _assert_turn(
            turn_1_expect, turn_1_resp, [], "model", "",
            _empty_stats(), _empty_stats(),
        )
        t2_failures = _assert_turn(
            turn_2_expect, turn_2_resp, [], "model", "",
            _empty_stats(), _empty_stats(),
        )

        assert t1_failures == [], f"Turn 1 should pass, got: {t1_failures}"
        assert t2_failures == [], f"Turn 2 should pass, got: {t2_failures}"

    def test_handoff_checked_per_turn(self) -> None:
        """Schema B: handoff is checked independently per turn."""
        turn_2_expect = {"handoff": False}
        turn_2_resp = _StubResponse(answer="We need to escalate this.", handoff=True)

        failures = _assert_turn(
            turn_2_expect, turn_2_resp, [], "model", "",
            _empty_stats(), _empty_stats(),
        )
        assert failures, "Expected handoff failure"
        assert "handoff" in failures[0]

    def test_citation_stats_accumulated_across_turns(self) -> None:
        """Schema B: citation stats should accumulate across turns."""
        citation_stats = _empty_stats()

        turn_1_expect = {"required_sources": ["01-returns-policy-current.md"]}
        turn_1_resp = _StubResponse(
            answer="Returns are 30 days.",
            citations=[{"filename": "01-returns-policy-current.md", "heading": ""}],
        )

        turn_2_expect = {"required_sources": ["10-gift-cards-and-price-adjustments.md"]}
        turn_2_resp = _StubResponse(
            answer="Price adjustments require human review.",
            citations=[],
        )

        _assert_turn(
            turn_1_expect, turn_1_resp, [], "model", "",
            citation_stats, _empty_stats(),
        )
        _assert_turn(
            turn_2_expect, turn_2_resp, [], "model", "",
            citation_stats, _empty_stats(),
        )

        assert citation_stats["total"] == 2, (
            f"Expected 2 total citations, got {citation_stats}"
        )
        assert citation_stats["hits"] == 1, (
            f"Expected 1 hit, got {citation_stats}"
        )
