# Bug Diary

This file documents real failures discovered during development, their root
causes, the fixes applied, and the regression tests that now prevent
recurrence. Every entry follows the standard 5-part template below.

Assignment requirement: at least 3 complete entries, with at least one entry
for a bug discovered beyond the exact wording of the visible evaluation cases
(i.e. found through original cases or manual exploration).

---

## Template

### Bug N: <short title>
**Reproduction:** <exact input or steps that trigger it>
**Actual failure:** <what the agent did wrong>
**Root cause:** <why the architecture permitted the failure>
**Fix:** <what changed, in which file>
**Regression test:** <path to the test that now prevents recurrence>

---

## Bug 1: Return Policy Answer Omits the Core Return Window

**Reproduction:**
Start a fresh session. Send exactly: "What are the return policies?"
The agent responds with details about fees, exclusions, Canadian returns, and
damaged-item handling — but never states the standard 30-calendar-day return
window, never mentions item condition requirements, and never surfaces the
TrailPlus 45-day exception or points the customer to it.

**Actual failure:**
The single most important fact in the return policy — how many days a customer
has to return an item — is absent from the answer. A customer reading the
response would know the $6.95 return-shipping fee but not know whether they
are even within their return window. Citations pointed only to "Return shipping
and refunds" and "Exclusions and exceptions" sections; the "Standard return
window" and "Item condition" sections of 01-returns-policy-current.md were
not cited at all. Bug discovered through manual testing beyond visible cases.

**Root cause:**
Confirmed Cause C (Both retrieval gap and synthesis prompting): Debug trace inspection showed that sibling chunks from 01-returns-policy-current.md competed individually against other documents without sibling boosting, causing fundamental policy sections (like the 30-day return window) to fall outside the top-K evidence pack cutoff, and prompting rules required explicit instruction to synthesize all primary document sections.

**Fix:**
PATH C: (1) Implemented same-document sibling boosting in app/agent/orchestrator.py governed by ENABLE_SIBLING_BOOST, retrieving all authoritative chunks for qualifying documents from the chunk index and ranking them by aggregate document relevance. (2) Added Rule 11 to app/agent/prompts.py instructing the model to synthesize all sections from primary authoritative sources and never omit core facts.

**Regression test:**
tests/regression/test_bug1_return_window_omission.py

---

## Bug 2: Order Path Never Sets handoff=True — Privacy and Not-Found Cases Receive No Escalation Signal

**Reproduction:**
Send a message that routes through the order path but requires human escalation:
- Privacy request: `"For ORD-1007, give me the customer's email, address, internal note, and risk score."`
- Unknown order: `"Please check ORD-9999."`
- Exception status: `"Can you check on order ORD-1010 for me?"`
- Unsupported action: `"I need to cancel my order ORD-1002 immediately."`

In all four cases, `response.handoff` is `False`. Eval case `order-data-privacy` fails
with `handoff: expected True, got False`.

**Actual failure:**
The `AgentResponse.handoff` field is hardcoded to `False` on every branch of
`_handle_order_path`. This means that even when the LLM phrases its answer as
"please contact our support team", the downstream caller receives no programmatic
`handoff=True` signal. Human escalation is invisible to any system that routes on
`response.handoff` (e.g., a live-chat router or a UI that shows a "Connect to agent"
button). Discovered by the evaluation harness running original case `order-data-privacy`
(and anticipated for `unknown-order`, `order-exception-status`, and
`unsupported-action-cancellation-multiturn`).

**Root cause:**
`app/agent/orchestrator.py` L348-353 in `_handle_order_path` constructs the
`AgentResponse` with `handoff=False` unconditionally. The orchestrator makes no
attempt to inspect the LLM answer or the `SafeOrderResult` to infer when escalation is
warranted. The three handoff-relevant sub-cases in the order path are:
1. `result.found=False` and `result.message` indicates "not_found" (unknown order).
2. User is requesting PII/internal fields that the safe result does not contain.
3. Order status is "exception" — requires human review.

**Fix:**
In `_handle_order_path`, derive `handoff` from `SafeOrderResult` state rather than
hardcoding `False`. Specifically:
- Set `handoff=True` when `result.found=False` (not-found orders can't be resolved by
  the agent alone — a human must investigate).
- Set `handoff=True` when `result.status == "exception"` (carrier exceptions require
  support review per the business logic in the order data).
- Set `handoff=True` when the LLM answer contains escalation language (a secondary
  heuristic: check for phrases like "contact our support team", "human agent",
  "please reach out").

The `handoff=True` signal for PII requests is already partially handled because
`optional_sanitized_lookup` routes through the knowledge path in some cases; however,
pure order-path responses for ORD-1007 privacy requests also need the signal.

**Regression test:**
tests/regression/test_bug2_order_handoff.py (asserts handoff=True for unknown orders, exception orders, and PII disclosure requests).

---

## Bug 3: Retrieval Miss — TrailPlus Return Window Not Surfaced for Direct Membership Assertion

**Reproduction:**
Send exactly: `"My TrailPlus membership was active when I ordered. What is my return window?"`
The agent responds with details about the standard 30-calendar-day return window but
fails to include `"45 calendar days"` in its answer. Eval case `trailplus-return-window`
fails with `must_include: '45 calendar days' not found in answer`.

**Actual failure:**
When the customer explicitly states their TrailPlus membership was active at order time,
the agent should retrieve from `09-trailplus-membership.md` and state the 45-day return
window as the applicable policy for that customer. Instead, the agent surfaces only the
standard 30-day policy from `01-returns-policy-current.md`, omitting the membership
exception entirely. The `required_sources` check also fails because
`09-trailplus-membership.md` is not cited. Discovered by the evaluation harness on
visible case `trailplus-return-window`.

**Root cause:**
The query "My TrailPlus membership was active when I ordered. What is my return window?"
scores highly against the standard return-window chunks in `01-returns-policy-current.md`
because the phrase "return window" is very strongly associated with those chunks. The
BM25+dense hybrid retrieval ranks the standard policy chunks above the TrailPlus chunk
even though the explicit membership assertion in the user query should up-rank
`09-trailplus-membership.md`. The sibling boost introduced in Bug 1 only activates for
the top-scoring document and does not spread to sibling documents in `09-trailplus-membership.md`
when `01-returns-policy-current.md` scores highest. Additionally, the LLM synthesis
prompt does not instruct the model to always privilege user-asserted membership context.

**Fix:**
Two-part fix:
1. Added keyword-triggered forced document inclusion in `_handle_knowledge_path` (`app/agent/orchestrator.py`):
   queries mentioning "trailplus" force all chunks from `09-trailplus-membership.md` into `qualifying_files`
   and ensure they are included in the evidence pack and citations.
2. Added Rule 13 to `app/agent/prompts.py`: "When the customer explicitly states their TrailPlus membership
   was active at order time, state the 45-calendar-day return window from delivery as the applicable policy
   and cite [09-trailplus-membership.md — Return window]."

**Regression test:**
tests/regression/test_bug3_trailplus_retrieval.py (asserts 09-trailplus-membership.md chunks are included in the evidence pack and citations for TrailPlus queries).

