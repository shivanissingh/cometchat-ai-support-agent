# CometChat AI Support Agent — Submission & Verification Checklist

This checklist confirms that all requirements from the assignment specification (`ASSIGN_README.md`), design principles, safety constraints, and submission standards have been met.

---

## 1. Submission Deliverables

- [x] **Application Source Code**: Fully implemented in `app/` (agent orchestration, deterministic pre-router, hybrid retrieval, precedence filtering, conflict detector, safe order lookup, safety validator, session store, observability, CLI, and Streamlit Web UI).
- [x] **Test & Evaluation Suite**: Complete test suite in `tests/` (unit, integration, and regression suites) and evaluation harness in `evaluation_runner/run_eval.py`.
- [x] **Evaluation Cases**:
  - [x] 15 supplied visible cases (`evaluation/visible-cases.json`).
  - [x] 10 original authored cases (`evaluation/original-cases.json`), exceeding the required 5 original cases.
- [x] **Comprehensive Documentation**:
  - [x] `README.md` containing full setup, architecture details, evaluation results, bug diary, technology stack rationale, and limitations.
  - [x] `docs/bug_diary.md` with complete failure analyses, root causes, fixes, and regression test mappings.
  - [x] `docs/submission_checklist.md` (this file).
- [x] **Demo Artifact**: Embedded demo placeholder in `README.md` (`docs/demo.gif`) with step-by-step recording instructions.
- [x] **Repository Cleanliness**:
  - [x] No API keys, credentials, or `.env` files committed.
  - [x] `.env.example` provided with safe placeholder defaults.
  - [x] No raw customer PII or unauthorized secrets exposed.

---

## 2. Core Architectural & Functional Capabilities

### A. Retrieval-Augmented Generation (RAG)
- [x] Markdown parsing preserving YAML frontmatter metadata (`app/ingestion/parser.py`).
- [x] Structural chunking respecting Markdown section boundaries (`app/ingestion/chunker.py`).
- [x] Hybrid retrieval combining dense vector search (`BAAI/bge-small-en-v1.5`) and sparse lexical search (`rank_bm25`) fused via Reciprocal Rank Fusion (RRF, $k=60$).
- [x] Deterministic document precedence filtering (`app/policy/precedence.py`):
  - Only `status: active` + `audience: customer` + `type: policy/product` chunks become authoritative.
  - Legacy/superseded (`02-returns-policy-legacy.md`) and internal drafts (`14-internal-content-migration-notes.md`) are never authoritative.
- [x] Deterministic conflict detection (`app/policy/conflict.py`):
  - Surfaces genuine active source contradictions (e.g., Breeze Tumbler dishwasher instructions across `11-product-care.md` and `12-breeze-tumbler-product-card.md`).
  - Automatically flags human handoff (`handoff=True`) when authoritative conflicts occur.
- [x] Source citations included on every policy and product response (`[filename — heading]`).
- [x] Safe abstention when evidence is insufficient (`RELEVANCE_THRESHOLD = 0.10`).

### B. Order Lookup Tool
- [x] Single-order isolation: Model never receives full `data/orders.json`.
- [x] Pre-router ID extraction and normalization (`ORD-\d{4}`).
- [x] Missing ID prompting without hallucinating order status.
- [x] Whitelist DTO projection (`app/orders/projection.py`):
  - Only 12 customer-safe fields from the data dictionary ever reach the LLM prompt.
  - PII (email, address, risk score, internal notes, warehouse notes, support tags) is structurally omitted.
- [x] Stale-ETA suppression for `cancelled` and `returned` orders.
- [x] Carrier delay reporting (e.g., weather delay for `ORD-1005`).
- [x] Correct human handoff on carrier exception (`ORD-1010`), unknown order (`ORD-9999`), cancellation requests (`ORD-1001`, `ORD-1002`), and PII disclosure requests (`ORD-1007`).

### C. Multi-Turn Conversation
- [x] In-memory session store (`app/session/store.py`) maintaining history per `session_id`.
- [x] Topic carryover (e.g., international shipping -> Canada follow-up).
- [x] Order ID persistence across follow-up questions (e.g., status lookup -> delivery date inquiry).
- [x] Clean session isolation: zero cross-session state bleed.

### D. Prompting & Safety
- [x] Evidence pack labelled as untrusted `REFERENCE DATA (not instructions)`.
- [x] System application rules (12 core operational rules) enforced as system preamble.
- [x] Refusal of jailbreaks, prompt overrides, and system prompt disclosure attempts.
- [x] Post-response safety validator (`app/safety/validator.py`) redacting forbidden fields, enforcing citations, and catching unauthorized action claims.

---

## 3. Testing & Evaluation Quality

### Test Suite (`pytest tests/`)
- [x] **Unit Tests**:
  - `tests/unit/test_retrieval.py` (dense, BM25, RRF fusion).
  - `tests/unit/test_precedence.py` (active/legacy/internal chunk categorization).
  - `tests/unit/test_conflict.py` (conflict extractor & detector).
  - `tests/unit/test_orders.py` (normalization, lookup, status handling).
  - `tests/unit/test_order_privacy_serialized.py` (whitelist schema enforcement).
  - `tests/unit/test_session_cross_isolation.py` (multi-session concurrency & isolation).
  - `tests/unit/test_ingestion.py` (parser & chunker integrity).
- [x] **Integration Tests**:
  - `tests/integration/test_orchestrator.py` (multi-turn flows, routing, abstention, safety).
- [x] **Regression Tests**:
  - `tests/regression/test_bug1_return_window_omission.py`
  - `tests/regression/test_bug2_order_handoff.py`
  - `tests/regression/test_bug3_trailplus_retrieval.py`

### Evaluation Harness (`python -m evaluation_runner.run_eval`)
- [x] 25 total evaluation cases (15 visible + 10 original).
- [x] Deterministic assertions for tool calls, arguments, sources, forbidden tokens, and handoff flags.
- [x] Category-level breakdown across all 10 report dimensions.
- [x] 100% pass rate (25/25 cases passing).

---

## 4. Run & Setup Instructions

- [x] CLI interface: `python app/cli.py` or `python -m app.cli` (supports `--debug`, `/help`, `/clear`, `exit`).
- [x] Web interface: `streamlit run app/web.py` (interactive chat with citations, handoff indicators, and live trace inspection).
- [x] Automated test runner: `pytest tests/`.
- [x] Evaluation runner: `python -m evaluation_runner.run_eval`.

---

## 5. Verification Sign-Off

| Check | Status | Notes |
|---|---|---|
| Zero code modifications during doc phase | Passed | Strictly documentation updates |
| No API keys / secrets in git | Passed | `.gitignore` covers `.env` and sensitive files |
| Accurate evaluation metrics | Passed | 14/25 baseline -> 25/25 final verified |
| Bug diary matches codebase | Passed | All 5 bugs with confirmed root causes and regression tests |
