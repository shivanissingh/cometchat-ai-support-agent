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

## Bug 2: [Title TBD]

[To be filled in when the next bug is found — either by the evaluation
harness (A5) or further manual testing.]

---

## Bug 3: [Title TBD]

[To be filled in when the next bug is found.]

