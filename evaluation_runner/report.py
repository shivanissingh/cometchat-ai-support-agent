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

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


@dataclass
class CaseResult:
    """Result of running a single evaluation case."""

    case_id: str
    category: str
    result: str  # "PASS", "FAIL", or "ERROR"
    reason: str  # One-line explanation


def print_report(
    results: list[CaseResult],
    citation_stats: dict,
    tool_arg_stats: dict,
    all_categories: list[str],
) -> None:
    """Print the formatted evaluation report and exit with appropriate code."""
    w = 80
    border = "═" * w

    print()
    print(f"{_BOLD}{_CYAN}╔{border}╗{_RESET}")
    print(f"{_BOLD}{_CYAN}║{'EVALUATION SUITE FINAL REPORT':^80}║{_RESET}")
    print(f"{_BOLD}{_CYAN}╚{border}╝{_RESET}")

    # --- Per-case table ---
    t_top = "┌" + "─" * 42 + "┬" + "─" * 16 + "┬" + "─" * 8 + "┬" + "─" * 26 + "┐"
    t_mid = "├" + "─" * 42 + "┼" + "─" * 16 + "┼" + "─" * 8 + "┼" + "─" * 26 + "┤"
    t_bot = "└" + "─" * 42 + "┴" + "─" * 16 + "┴" + "─" * 8 + "┴" + "─" * 26 + "┘"

    print(f"\n{_BOLD}{t_top}{_RESET}")
    hdr_cases = (
        f"{_BOLD}│ {'CASE ID':<40} │ {'CATEGORY':<14} │ {'RESULT':<6} │ "
        f"{'DETAILS / REASON':<24} │{_RESET}"
    )
    print(hdr_cases)
    print(f"{_BOLD}{t_mid}{_RESET}")

    for r in results:
        cid = r.case_id[:40]
        cat = r.category[:14]
        if r.result == "PASS":
            res_str = f"{_GREEN}{_BOLD}PASS{_RESET}  "
            reason_str = f"{_GREEN}{r.reason[:24]:<24}{_RESET}"
        elif r.result == "FAIL":
            res_str = f"{_RED}{_BOLD}FAIL{_RESET}  "
            reason_str = f"{_RED}{r.reason[:24]:<24}{_RESET}"
        else:
            res_str = f"{_YELLOW}{_BOLD}ERR {_RESET}  "
            reason_str = f"{_YELLOW}{r.reason[:24]:<24}{_RESET}"

        print(f"│ {cid:<40} │ {cat:<14} │ {res_str} │ {reason_str} │")

    print(f"{_BOLD}{t_bot}{_RESET}")

    # --- Per-category summary ---
    c_top = "┌" + "─" * 28 + "┬" + "─" * 14 + "┬" + "─" * 8 + "┬" + "─" * 14 + "┐"
    c_mid = "├" + "─" * 28 + "┼" + "─" * 14 + "┼" + "─" * 8 + "┼" + "─" * 14 + "┤"
    c_bot = "└" + "─" * 28 + "┴" + "─" * 14 + "┴" + "─" * 8 + "┴" + "─" * 14 + "┘"

    print(f"\n{_BOLD}{_BLUE}{c_top}{_RESET}")
    hdr_cat = (
        f"{_BOLD}{_BLUE}│ {'Category':<26} │ {'Pass / Total':<12} │ "
        f"{'Rate':<6} │ {'Status':<12} │{_RESET}"
    )
    print(hdr_cat)
    print(f"{_BOLD}{_BLUE}{c_mid}{_RESET}")

    category_pass: dict[str, int] = {cat: 0 for cat in all_categories}
    category_total: dict[str, int] = {cat: 0 for cat in all_categories}

    for r in results:
        cat = r.category
        if cat in category_total:
            category_total[cat] += 1
            if r.result == "PASS":
                category_pass[cat] += 1

    c_hits = citation_stats.get("hits", 0)
    c_total = citation_stats.get("total", 0)
    ta_hits = tool_arg_stats.get("hits", 0)
    ta_total = tool_arg_stats.get("total", 0)

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

        ratio_str = f"{passed}/{total}"
        if total > 0:
            pct_val = (100 * passed) // total
            pct_str = f"{pct_val}%"
            if pct_val == 100:
                status_str = f"{_GREEN}✓ COMPLETE{_RESET}  "
            elif pct_val >= 70:
                status_str = f"{_YELLOW}⚡ PARTIAL{_RESET}   "
            else:
                status_str = f"{_RED}✗ ATTENTION{_RESET} "
        else:
            pct_str = "N/A"
            status_str = f"{_DIM}- SKIPPED{_RESET}  "

        print(f"│ {cat:<28} │ {ratio_str:<12} │ {pct_str:<6} │ {status_str} │")

    print(f"{_BOLD}{_BLUE}{c_bot}{_RESET}")

    # --- Overall ---
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r.result == "PASS")
    failed_cases = sum(1 for r in results if r.result == "FAIL")
    error_cases = sum(1 for r in results if r.result == "ERROR")

    print()
    if failed_cases == 0 and error_cases == 0:
        print(f"{_GREEN}{_BOLD}╔{border}╗{_RESET}")
        msg = f"🎉 ALL EVALUATION CASES PASSED! ({passed_cases}/{total_cases})"
        print(f"{_GREEN}{_BOLD}║{msg:^80}║{_RESET}")
        print(f"{_GREEN}{_BOLD}╚{border}╝{_RESET}\n")
    else:
        print(f"{_RED}{_BOLD}╔{border}╗{_RESET}")
        msg1 = f"OVERALL RESULTS: {passed_cases}/{total_cases} CASES PASSED"
        msg2 = f"Passed: {passed_cases} | Failed: {failed_cases} | Errors: {error_cases}"
        print(f"{_RED}{_BOLD}║{msg1:^80}║{_RESET}")
        print(f"{_RED}{_BOLD}║{msg2:^80}║{_RESET}")
        print(f"{_RED}{_BOLD}╚{border}╝{_RESET}\n")

    if failed_cases > 0 or error_cases > 0:
        sys.exit(1)
    else:
        sys.exit(0)
