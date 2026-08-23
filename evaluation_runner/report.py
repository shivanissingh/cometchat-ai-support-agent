"""
evaluation_runner/report.py - Reporting module for the evaluation harness.

Produces:
  1. Per-case table:  case_id | category | result | reason (one line max).
  2. Per-category summary table: pass/total for each of the 10 report categories.
  3. Overall: X/25 cases passed.

Exit code: 0 if all pass, non-zero if any FAIL or ERROR.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class CaseResult:
    """Result of running a single evaluation case."""

    case_id: str
    category: str
    result: str   # "PASS", "FAIL", or "ERROR"
    reason: str   # One-line explanation


def print_report(
    results: list[CaseResult],
    citation_stats: dict,
    tool_arg_stats: dict,
    all_categories: list[str],
) -> None:
    """Print the full evaluation report and exit with appropriate code.

    Parameters
    ----------
    results:
        One CaseResult per case, in run order.
    citation_stats:
        {"hits": int, "total": int} - cross-case citation assertion counts.
    tool_arg_stats:
        {"hits": int, "total": int} - cross-case tool argument assertion counts.
    all_categories:
        Complete list of report categories (10) that must appear in the summary.
    """
    print()
    print("=" * 90)
    print("EVALUATION REPORT")
    print("=" * 90)

    # --- Per-case table ---
    print()
    print(f"{'CASE ID':<55} {'CATEGORY':<22} {'RESULT':<7} REASON")
    print("-" * 90)

    for r in results:
        reason_trunc = r.reason[:35] if len(r.reason) > 35 else r.reason
        result_icon = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR "}.get(r.result, r.result)
        print(
            f"{r.case_id:<55} {r.category:<22} {result_icon:<7} {reason_trunc}"
        )

    # --- Per-category summary ---
    print()
    print("=" * 60)
    print("PER-CATEGORY SUMMARY")
    print("=" * 60)

    # Count pass/total by category (based on case results).
    category_pass: dict[str, int] = {cat: 0 for cat in all_categories}
    category_total: dict[str, int] = {cat: 0 for cat in all_categories}

    for r in results:
        cat = r.category
        if cat in category_total:
            category_total[cat] += 1
            if r.result == "PASS":
                category_pass[cat] += 1

    # Inject Citation and Tool arguments stats (they are per-assertion, not per-case).
    # They are shown as hits/total assertions rather than cases.
    c_hits = citation_stats.get("hits", 0)
    c_total = citation_stats.get("total", 0)
    ta_hits = tool_arg_stats.get("hits", 0)
    ta_total = tool_arg_stats.get("total", 0)

    print(f"{'Category':<25} {'Pass/Total':<12} {'%'}")
    print("-" * 50)
    for cat in all_categories:
        if cat == "Citation":
            total = c_total
            passed = c_hits
        elif cat == "Tool arguments":
            total = ta_total
            passed = ta_hits
        else:
            total = category_total.get(cat, 0)
            passed = category_pass.get(cat, 0)

        pct = f"{100 * passed // total}%" if total > 0 else "N/A"
        print(f"  {cat:<23} {passed}/{total:<11} {pct}")

    # --- Overall ---
    print()
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r.result == "PASS")
    failed_cases = sum(1 for r in results if r.result == "FAIL")
    error_cases = sum(1 for r in results if r.result == "ERROR")

    print("=" * 60)
    print(f"OVERALL: {passed_cases}/{total_cases} cases passed")
    if failed_cases:
        print(f"  FAIL:  {failed_cases} cases")
    if error_cases:
        print(f"  ERROR: {error_cases} cases")
    print("=" * 60)
    print()

    if failed_cases > 0 or error_cases > 0:
        sys.exit(1)
    else:
        sys.exit(0)
