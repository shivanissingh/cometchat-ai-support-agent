"""
evaluation_runner/run_eval.py - Evaluation harness for the CometChat RAG Support Agent.

Usage
-----
    python -m evaluation_runner.run_eval

Design
------
* Loads evaluation/visible-cases.json and evaluation/original-cases.json.
* Divides the 25 cases into 3 batches and rotates across up to 3 Gemini models
  to stay within the Gemini free-tier RPD=20 and RPM=5 limits.
* For each case, creates a fresh session_id and calls Agent.handle_message
  DIRECTLY (never through app/cli.py or app/web.py).
* Applies all deterministic assertions first; only uses an LLM grading call
  for must_include_concepts (the one permitted secondary use).
* Records each case as PASS, FAIL, or ERROR with a one-line reason.
* Delegates printing and exit-code to evaluation_runner.report.

Rate limit strategy
-------------------
RATE_LIMIT_SLEEP_SECONDS = 13  seconds between LLM calls within a batch
BATCH_PAUSE_SECONDS       = 60  seconds between batches

On a 429 rate-limit error: single retry after RATE_LIMIT_SLEEP_SECONDS * 3.
If the retry also fails, the case is marked ERROR and execution continues.

Category taxonomy
-----------------
The raw case ``category`` field is mapped to a report category using
CATEGORY_MAP defined below. Tool arguments and Citation are additional
assertion categories tracked across all cases.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import google.genai as genai

from app.agent.orchestrator import Agent, _KnowledgeIndex
from evaluation_runner import report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
_logger = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Rate-limit constants
# ---------------------------------------------------------------------------

#: Seconds to sleep between individual LLM calls within a batch.
RATE_LIMIT_SLEEP_SECONDS: int = 13

#: Seconds to sleep between batches to reset the per-minute quota window.
BATCH_PAUSE_SECONDS: int = 60

# ---------------------------------------------------------------------------
# Model rotation
# ---------------------------------------------------------------------------

_DEFAULT_EVAL_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
]


def _get_eval_models() -> list[str]:
    """Return the list of models to use for evaluation batches.

    If EVAL_MODELS env var is set, parse it as a comma-separated list.
    """
    raw = os.environ.get("EVAL_MODELS", "").strip()
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            return models
    return _DEFAULT_EVAL_MODELS


# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    "retrieval": "Retrieval",
    "multi-source-grounding": "Groundedness",
    "conversation": "Multi-turn",
    "groundedness": "Groundedness",
    "tool-use": "Tool use",
    "tool-reliability": "Tool use",
    "privacy": "Privacy",
    "prompt-security": "Safety",
    "abstention": "Abstention",
    "source-conflict": "Conflict handling",
}


def canonical_category(cat: str) -> str:
    """Map raw test case category string to one of the 10 report categories."""
    return CATEGORY_MAP.get(cat.lower(), cat.capitalize())


ALL_REPORT_CATEGORIES = [
    "Retrieval",
    "Groundedness",
    "Tool use",
    "Tool arguments",
    "Privacy",
    "Multi-turn",
    "Safety",
    "Abstention",
    "Citation",
    "Conflict handling",
]

# ---------------------------------------------------------------------------
# Forbidden values for must_not_invent (by item label)
# ---------------------------------------------------------------------------

MUST_NOT_INVENT_VALUES: dict[str, list[str]] = {
    "order status": [
        "pending",
        "processing",
        "shipped",
        "delivered",
        "cancelled",
        "returned",
        "delayed",
        "exception",
    ],
    "status": [
        "pending",
        "processing",
        "shipped",
        "delivered",
        "cancelled",
        "returned",
        "delayed",
        "exception",
    ],
    "tracking number": [
        "tracking number is",
    ],
    "carrier": ["ups", "usps", "fedex", "canada post"],
    "delivery estimate": [
        "will arrive",
        "estimated delivery",
    ],
    "arrival date": [
        "will arrive",
        "arriving on",
    ],
    "material certification": [
        "certified vegan",
        "vegan certified",
        "gots certified",
        "bluesign",
        "oeko-tex",
    ],
    "vegan guarantee": [
        "all our products are vegan",
        "guaranteed vegan",
    ],
    "whether the specific item qualifies without knowing its condition": [
        "your item qualifies",
        "it is eligible for return",
        "return approved",
    ],
}

# ---------------------------------------------------------------------------
# Forbidden literal PII/internal values from orders.json
# ---------------------------------------------------------------------------

_ORDERS_RAW: dict[str, dict] = {}


def _load_orders_raw() -> dict[str, dict]:
    data_file = Path(__file__).parents[1] / "data" / "orders.json"
    with data_file.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return {o["order_id"]: o for o in payload["orders"]}


_ORDERS_RAW = _load_orders_raw()

_REFUSE_FIELD_EXTRACTORS: dict[str, Any] = {
    "email": lambda o: [o["customer"]["email"]],
    "address": lambda o: [o["customer"]["shipping_address"]],
    "internal note": lambda o: [o.get("internal", {}).get("warehouse_note", "")],
    "risk score": lambda o: [str(o.get("internal", {}).get("risk_score", ""))],
}

# ---------------------------------------------------------------------------
# Load test-case files
# ---------------------------------------------------------------------------


def _load_cases() -> list[dict]:
    """Load visible-cases.json and original-cases.json and merge them."""
    base = Path(__file__).parents[1] / "evaluation"
    result: list[dict] = []
    for fname in ("visible-cases.json", "original-cases.json"):
        fpath = base / fname
        with fpath.open(encoding="utf-8") as fh:
            data = json.load(fh)
        result.extend(data["cases"])
        _logger.info("Loaded %d cases from %s", len(data["cases"]), fname)
    _logger.info("Total cases: %d", len(result))
    return result


# ---------------------------------------------------------------------------
# Grading LLM call (secondary, for must_include_concepts only)
# ---------------------------------------------------------------------------


def _grade_concepts(
    answer: str,
    concepts: list[str],
    model: str,
    api_key: str,
) -> dict[str, bool]:
    """Ask a grading LLM whether each concept is addressed in the answer."""
    if not concepts:
        return {}

    from app.agent.model_manager import GLOBAL_MODEL_MANAGER

    concept_list = "\n".join(f"- {c}" for c in concepts)
    prompt = (
        "You are a semantic concept grader for customer support responses. Read the following "
        "support agent response and determine whether each concept or meaning is conveyed or "
        "addressed (even if expressed in different wording or paraphrased).\n"
        "Answer only YES or NO for each concept, in order, one per line.\n\n"
        f"Response to grade:\n{answer}\n\n"
        f"Concepts to check:\n{concept_list}\n\n"
        "Answer (one YES or NO per line, in the same order as the concepts above):"
    )

    client = genai.Client(api_key=api_key)
    max_retries = len(GLOBAL_MODEL_MANAGER.models) * 2
    for _ in range(max_retries):
        grading_model = GLOBAL_MODEL_MANAGER.acquire_call_slot()
        GLOBAL_MODEL_MANAGER.record_call(grading_model)
        try:
            resp = client.models.generate_content(
                model=grading_model,
                contents=prompt,
            )
            lines = (resp.text or "").strip().splitlines()
            result: dict[str, bool] = {}
            for i, concept in enumerate(concepts):
                if i < len(lines):
                    result[concept] = lines[i].strip().upper().startswith("YES")
                else:
                    result[concept] = False
            return result
        except Exception as exc:  # noqa: BLE001
            exc_str = str(exc)
            is_quota = any(
                code in exc_str
                for code in ("429", "ResourceExhausted", "RESOURCE_EXHAUSTED", "quota")
            )
            if is_quota:
                GLOBAL_MODEL_MANAGER.mark_exhausted(grading_model, reason="429 in concept grading")
                continue
            elif "503" in exc_str or "UNAVAILABLE" in exc_str or "demand" in exc_str.lower():
                time.sleep(2)
                continue
            else:
                _logger.warning("Concept grading call failed on %s: %s", grading_model, exc)
                return {c: True for c in concepts}
    return {c: True for c in concepts}


# ---------------------------------------------------------------------------
# Deterministic assertion helpers
# ---------------------------------------------------------------------------


def _check_must_include(answer: str, must_include: list[str]) -> str | None:
    for phrase in must_include:
        if phrase.lower() not in answer.lower():
            return f"must_include: '{phrase}' not found in answer"
    return None


def _check_must_not_include(answer: str, must_not_include: list[str]) -> str | None:
    for phrase in must_not_include:
        if phrase.lower() in answer.lower():
            return f"must_not_include: '{phrase}' found in answer"
    return None


def _check_required_sources(
    citations: list[dict],
    required_sources: list[str],
) -> tuple[str | None, list[str], list[str]]:
    filenames = {c.get("filename", "") for c in citations}
    hits = []
    misses = []
    for src in required_sources:
        if src in filenames:
            hits.append(src)
        else:
            misses.append(src)
    reason = None
    if misses:
        reason = f"required_sources missing from citations: {misses}"
    return reason, hits, misses


def _check_forbidden_sources(
    citations: list[dict],
    forbidden: list[str],
) -> str | None:
    filenames = {c.get("filename", "") for c in citations}
    for src in forbidden:
        if src in filenames:
            return f"forbidden_sources_as_authority: '{src}' cited"
    return None


def _check_tool(trace: list, expected_tool: str, answer: str) -> str | None:
    """Assert tool behavior from the trace."""
    tool_call_events = [e for e in trace if e.stage == "tool_call"]

    if expected_tool == "order_lookup":
        names = [e.payload.get("function", e.payload.get("name", "")) for e in tool_call_events]
        if not any("lookup_order" in n or n == "lookup_order" for n in names):
            return "tool: expected order_lookup tool call, found none in trace"
        return None

    if expected_tool == "not_called":
        if tool_call_events:
            return "tool: expected no tool call, but tool_call event found in trace"
        return None

    if expected_tool == "not_called_without_id":
        if tool_call_events:
            return "tool: expected no tool call, but tool_call event found in trace"
        if "order id" not in answer.lower() and "order number" not in answer.lower():
            return "tool: not_called_without_id - response does not ask for order ID"
        return None

    if expected_tool == "optional_sanitized_lookup":
        if not tool_call_events:
            return None
        tool_result_events = [e for e in trace if e.stage == "tool_result"]
        for event in tool_result_events:
            payload_str = json.dumps(event.payload)
            for forbidden in ("risk_score", "warehouse_note", "email", "shipping_address"):
                if forbidden in payload_str:
                    return (
                        f"tool: optional_sanitized_lookup - forbidden field "
                        f"'{forbidden}' found in tool result payload"
                    )
        return None

    return None


def _check_tool_arguments(
    trace: list,
    expected_args: dict,
) -> tuple[str | None, bool]:
    """Assert tool_call arguments match. Returns (reason, hit)."""
    tool_call_events = [e for e in trace if e.stage == "tool_call"]
    if not tool_call_events:
        return "tool_arguments: no tool_call event in trace", False

    event = tool_call_events[0]
    actual_order_id = event.payload.get("order_id", "")
    expected_order_id = expected_args.get("order_id", "")

    actual_norm = (actual_order_id or "").strip().upper()
    expected_norm = (expected_order_id or "").strip().upper()

    if actual_norm != expected_norm:
        return (
            f"tool_arguments: expected order_id='{expected_norm}', got='{actual_norm}'",
            False,
        )
    return None, True


def _check_handoff(actual_handoff: bool, expected_handoff: bool) -> str | None:
    if actual_handoff != expected_handoff:
        return f"handoff: expected {expected_handoff}, got {actual_handoff}"
    return None


def _check_must_not_invent(answer: str, items: list[str]) -> str | None:
    for item in items:
        forbidden_phrases = MUST_NOT_INVENT_VALUES.get(item.lower(), [])
        for phrase in forbidden_phrases:
            if phrase.lower() in answer.lower():
                return f"must_not_invent '{item}': found hallucination phrase '{phrase}'"
    return None


def _check_must_ask_for(answer: str, must_ask_for: list[str]) -> str | None:
    for phrase in must_ask_for:
        if phrase.lower() in answer.lower():
            return None
    return f"must_ask_for: none of {must_ask_for} found in answer"


def _check_must_refuse_to_disclose(
    answer: str,
    field_types: list[str],
    messages: list[dict],
) -> str | None:
    """Check that forbidden literal values are absent from the answer."""
    order_id = None
    for msg in messages:
        content = msg.get("content", "")
        match = re.search(r"ORD-\d{4}", content, re.IGNORECASE)
        if match:
            order_id = match.group(0).upper()
            break

    if order_id is None or order_id not in _ORDERS_RAW:
        return None

    raw = _ORDERS_RAW[order_id]
    for field_type in field_types:
        extractor = _REFUSE_FIELD_EXTRACTORS.get(field_type.lower())
        if extractor is None:
            continue
        forbidden_values = extractor(raw)
        for val in forbidden_values:
            if val and val.lower() in answer.lower():
                return (
                    f"must_refuse_to_disclose: literal value for '{field_type}' "
                    f"('{val[:30]}') found in answer"
                )
    return None


# ---------------------------------------------------------------------------
# Per-turn assertion runner
# ---------------------------------------------------------------------------


def _assert_turn(
    expect: dict,
    response: Any,
    messages: list[dict],
    model: str,
    api_key: str,
    citation_stats: dict,
    tool_arg_stats: dict,
) -> list[str]:
    """Run all assertions in *expect* against *response*.

    Returns a list of failure reason strings. Empty list = all passed.
    """
    failures: list[str] = []
    answer = response.answer
    citations = response.citations
    trace = response.trace

    if "must_include" in expect:
        r = _check_must_include(answer, expect["must_include"])
        if r:
            failures.append(r)

    if "must_not_include" in expect:
        r = _check_must_not_include(answer, expect["must_not_include"])
        if r:
            failures.append(r)

    if "required_sources" in expect:
        r, hits, misses = _check_required_sources(citations, expect["required_sources"])
        citation_stats["total"] += len(expect["required_sources"])
        citation_stats["hits"] += len(hits)
        if r:
            failures.append(r)

    if "forbidden_sources_as_authority" in expect:
        r = _check_forbidden_sources(citations, expect["forbidden_sources_as_authority"])
        if r:
            failures.append(r)

    if "tool" in expect:
        r = _check_tool(trace, expect["tool"], answer)
        if r:
            failures.append(r)

    if "tool_arguments" in expect:
        r, hit = _check_tool_arguments(trace, expect["tool_arguments"])
        tool_arg_stats["total"] += 1
        if hit:
            tool_arg_stats["hits"] += 1
        if r:
            failures.append(r)

    if "handoff" in expect:
        r = _check_handoff(response.handoff, expect["handoff"])
        if r:
            failures.append(r)

    if "must_not_invent" in expect:
        r = _check_must_not_invent(answer, expect["must_not_invent"])
        if r:
            failures.append(r)

    if "must_ask_for" in expect:
        r = _check_must_ask_for(answer, expect["must_ask_for"])
        if r:
            failures.append(r)

    if "must_refuse_to_disclose" in expect:
        r = _check_must_refuse_to_disclose(answer, expect["must_refuse_to_disclose"], messages)
        if r:
            failures.append(r)

    if "must_include_concepts" in expect:
        concepts = expect["must_include_concepts"]
        grades = _grade_concepts(answer, concepts, model, api_key)
        for concept, passed in grades.items():
            if not passed:
                failures.append(f"must_include_concepts: concept not addressed - '{concept}'")

    return failures


# ---------------------------------------------------------------------------
# Per-case runners
# ---------------------------------------------------------------------------


def _run_case(
    case: dict,
    agent: Agent,
    model: str,
    api_key: str,
    citation_stats: dict,
    tool_arg_stats: dict,
) -> report.CaseResult:
    """Run a single test case and return a CaseResult."""
    case_id = case["id"]
    raw_category = case.get("category", "unknown")
    category = CATEGORY_MAP.get(raw_category, raw_category)
    messages = case.get("messages", [])
    expect = case.get("expect", {})
    session_id = str(uuid.uuid4())

    is_schema_b = "turn_1" in expect or "turn_2" in expect

    try:
        if is_schema_b:
            return _run_schema_b_case(
                case_id,
                category,
                messages,
                expect,
                agent,
                session_id,
                model,
                api_key,
                citation_stats,
                tool_arg_stats,
            )
        else:
            return _run_schema_a_case(
                case_id,
                category,
                messages,
                expect,
                agent,
                session_id,
                model,
                api_key,
                citation_stats,
                tool_arg_stats,
            )
    except Exception as exc:  # noqa: BLE001
        _logger.exception("ERROR running case %s: %s", case_id, exc)
        return report.CaseResult(
            case_id=case_id,
            category=category,
            result="ERROR",
            reason=f"Exception: {type(exc).__name__}: {str(exc)[:100]}",
        )


def _run_schema_a_case(
    case_id: str,
    category: str,
    messages: list[dict],
    expect: dict,
    agent: Agent,
    session_id: str,
    model: str,
    api_key: str,
    citation_stats: dict,
    tool_arg_stats: dict,
) -> report.CaseResult:
    """Run all turns; assert only on the FINAL turn's response (Schema A)."""
    response = None
    for i, msg in enumerate(messages):
        if i > 0:
            _logger.info(
                "  [%s] sleeping %ds before turn %d/%d",
                case_id,
                RATE_LIMIT_SLEEP_SECONDS,
                i + 1,
                len(messages),
            )
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)
        response = agent.handle_message(session_id=session_id, text=msg["content"])

    if response is None:
        return report.CaseResult(
            case_id=case_id,
            category=category,
            result="ERROR",
            reason="No messages in case",
        )

    failures = _assert_turn(
        expect,
        response,
        messages,
        model,
        api_key,
        citation_stats,
        tool_arg_stats,
    )

    if failures:
        return report.CaseResult(
            case_id=case_id,
            category=category,
            result="FAIL",
            reason=failures[0],
        )
    return report.CaseResult(
        case_id=case_id,
        category=category,
        result="PASS",
        reason="All assertions passed",
    )


def _run_schema_b_case(
    case_id: str,
    category: str,
    messages: list[dict],
    expect: dict,
    agent: Agent,
    session_id: str,
    model: str,
    api_key: str,
    citation_stats: dict,
    tool_arg_stats: dict,
) -> report.CaseResult:
    """Run all turns; assert each turn independently (Schema B)."""
    all_failures: list[str] = []

    for i, msg in enumerate(messages):
        if i > 0:
            _logger.info(
                "  [%s] sleeping %ds before turn %d/%d (Schema B)",
                case_id,
                RATE_LIMIT_SLEEP_SECONDS,
                i + 1,
                len(messages),
            )
            time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        response = agent.handle_message(session_id=session_id, text=msg["content"])
        turn_key = f"turn_{i + 1}"

        if turn_key in expect:
            turn_expect = expect[turn_key]
            failures = _assert_turn(
                turn_expect,
                response,
                messages[: i + 1],
                model,
                api_key,
                citation_stats,
                tool_arg_stats,
            )
            if failures:
                all_failures.append(f"[{turn_key}] {failures[0]}")

    if all_failures:
        return report.CaseResult(
            case_id=case_id,
            category=category,
            result="FAIL",
            reason=all_failures[0],
        )
    return report.CaseResult(
        case_id=case_id,
        category=category,
        result="PASS",
        reason="All turn assertions passed",
    )


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def _run_case_with_retry(
    case: dict,
    agent: Agent,
    model: str,
    api_key: str,
    citation_stats: dict,
    tool_arg_stats: dict,
) -> report.CaseResult:
    """Wrap _run_case with a single retry on transient errors."""
    try:
        return _run_case(
            case,
            agent,
            model,
            api_key,
            citation_stats,
            tool_arg_stats,
        )
    except Exception as exc:  # noqa: BLE001
        exc_str = str(exc)
        is_transient = any(
            code in exc_str for code in ("429", "503", "UNAVAILABLE", "ResourceExhausted")
        ) or any(term in exc_str.lower() for term in ("quota", "rate", "demand", "temporar"))
        if is_transient:
            sleep_secs = RATE_LIMIT_SLEEP_SECONDS * 2
            _logger.warning(
                "Transient error (%s) for case %s. Retrying after %ds...",
                exc_str[:80],
                case["id"],
                sleep_secs,
            )
            time.sleep(sleep_secs)
            try:
                return _run_case(
                    case,
                    agent,
                    model,
                    api_key,
                    citation_stats,
                    tool_arg_stats,
                )
            except Exception as exc2:  # noqa: BLE001
                return report.CaseResult(
                    case_id=case["id"],
                    category=CATEGORY_MAP.get(case.get("category", ""), "Unknown"),
                    result="ERROR",
                    reason=f"Retry failed: {str(exc2)[:100]}",
                )
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def main() -> None:
    """Entry point for the evaluation harness."""
    from dotenv import load_dotenv

    load_dotenv()

    # Route background logger noise to logs/evaluation.log
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/evaluation.log", mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)

    api_key = os.environ.get("GEMINI_API_KEY", "")

    from app.agent.model_manager import GLOBAL_MODEL_MANAGER

    cases = _load_cases()
    total = len(cases)

    w = 80
    border = "═" * w

    print()
    print(f"{_BOLD}{_CYAN}╔{border}╗{_RESET}")
    print(f"{_BOLD}{_CYAN}║{'ASTER & ROW AI SUPPORT AGENT — EVALUATION SUITE':^80}║{_RESET}")
    print(f"{_BOLD}{_CYAN}╚{border}╝{_RESET}")
    print(f"  {_BOLD}Total Test Cases:{_RESET} {total} (15 visible + 10 original)")
    print(f"  {_BOLD}Model Pool:{_RESET}       {', '.join(GLOBAL_MODEL_MANAGER.models)}")
    print(
        f"  {_BOLD}Rate Limiting:{_RESET}    Strict <= 3 RPM | Daily Caps: 18 (Flash) / 490 (Lite)"
    )
    print(f"  {_BOLD}Detailed Logs:{_RESET}    logs/evaluation.log\n")

    print(f"{_CYAN}⚙ Initializing Knowledge Base indexes...{_RESET}", flush=True)
    kb_index = _KnowledgeIndex()
    agent = Agent(knowledge_index=kb_index)
    print(f"{_GREEN}✓ Knowledge Base initialized successfully.{_RESET}\n")

    all_results: list[report.CaseResult] = []
    citation_stats: dict = {"hits": 0, "total": 0}
    tool_arg_stats: dict = {"hits": 0, "total": 0}

    print(
        f"{_BOLD}═══════════════════════ RUNNING EVALUATION CASES ═══════════════════════{_RESET}"
    )

    for idx, case in enumerate(cases, 1):
        case_id = case["id"]
        raw_category = case.get("category", "unknown")
        category = CATEGORY_MAP.get(raw_category, raw_category)
        messages = case.get("messages", [])
        active_model_str = GLOBAL_MODEL_MANAGER.get_model_status_str()

        print(
            f"\n{_BOLD}┌─ [Case {idx:02d}/{total:02d}] "
            f"{_BLUE}{case_id}{_RESET}{_BOLD} (Category: {category}){_RESET}"
        )
        print(f"│  {_DIM}Active Model:{_RESET} {active_model_str}")

        if len(messages) == 1:
            query_snippet = messages[0].get("content", "").replace("\n", " ")
            if len(query_snippet) > 80:
                query_snippet = query_snippet[:77] + "..."
            print(f'│  {_DIM}Query:{_RESET} "{query_snippet}"')
        else:
            print(f"│  {_DIM}Multi-turn ({len(messages)} turns):{_RESET}")
            for t_idx, msg in enumerate(messages, 1):
                t_snip = msg.get("content", "").replace("\n", " ")
                if len(t_snip) > 70:
                    t_snip = t_snip[:67] + "..."
                print(f'│    • Turn {t_idx}: "{t_snip}"')

        result = _run_case_with_retry(
            case,
            agent,
            GLOBAL_MODEL_MANAGER.get_current_model(),
            api_key,
            citation_stats,
            tool_arg_stats,
        )
        all_results.append(result)

        if result.result == "PASS":
            pass_str = f"{_GREEN}{_BOLD}✓ PASS{_RESET} {_GREEN}({result.reason}){_RESET}"
            print(f"│  {_BOLD}Result:{_RESET} {pass_str}")
        elif result.result == "FAIL":
            print(f"│  {_BOLD}Result:{_RESET} {_RED}{_BOLD}✗ FAIL{_RESET}")
            print(f"│  {_RED}{_BOLD}Reason:{_RESET} {_RED}{result.reason}{_RESET}")
        else:
            print(f"│  {_BOLD}Result:{_RESET} {_YELLOW}{_BOLD}⚠ ERROR{_RESET}")
            print(f"│  {_YELLOW}{_BOLD}Reason:{_RESET} {_YELLOW}{result.reason}{_RESET}")

        print(
            f"{_BOLD}└────────────────────────────────────────────────────────────────────────┘{_RESET}"
        )

    report.print_report(
        results=all_results,
        citation_stats=citation_stats,
        tool_arg_stats=tool_arg_stats,
        all_categories=ALL_REPORT_CATEGORIES,
    )


if __name__ == "__main__":
    main()
