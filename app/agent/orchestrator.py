"""
app/agent/orchestrator.py — Single entrypoint for the support agent.

Usage
-----
::

    from app.agent.orchestrator import Agent, AgentResponse
    agent = Agent()
    response = agent.handle_message(session_id="s1", text="Where is ORD-1007?")
    print(response.answer)

Flow
----
1. Session store lookup → recent turns, last order_id, last topic.
2. Deterministic router decision (no LLM involved yet).
3. Branch on router path:
   a. order_ask_id   → deterministic "please provide order ID" response.
   b. order_direct / order_followup → order tool → evidence pack → LLM.
   c. knowledge      → retrieval → precedence → conflict → evidence pack.
      - Zero auth chunks above RELEVANCE_THRESHOLD → deterministic abstention.
      - Conflict detected → evidence pack + LLM + handoff=True.
      - Normal → LLM → check for function-call fallback.
4. Safety validator (post-processing).
5. Session store update.
6. Return AgentResponse.

Named constants
---------------
RELEVANCE_THRESHOLD:
    Minimum final_score an authoritative chunk must achieve to be included
    in the evidence pack and trigger a real LLM call.  Chunks below this
    threshold are treated as "no evidence found", producing a deterministic
    abstention with handoff=True.

ABSTENTION_RESPONSE:
    The fixed text returned when no authoritative evidence is found.

ORDER_ASK_ID_RESPONSE:
    The fixed text returned when order-intent is detected but no ID supplied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.agent import llm_client
from app.agent.prompts import APP_RULES_TEXT, format_evidence_pack
from app.agent.router import RouterDecision, route
from app.ingestion.chunker import build_chunk_index, chunk_all_documents
from app.ingestion.parser import parse_all
from app.observability import emit_trace
from app.orders.lookup import lookup_order
from app.policy.conflict import ConflictDetector
from app.policy.precedence import mark_authoritative
from app.retrieval.bm25_index import BM25Index
from app.retrieval.dense_index import DenseIndex
from app.retrieval.fusion import reciprocal_rank_fusion
from app.safety.validator import validate_response
from app.schemas import ConflictResult, RetrievedChunk, SafeOrderResult, TraceEvent
from app.session.store import SESSION_STORE, Turn

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants (tune against the evaluation suite)
# ---------------------------------------------------------------------------

#: Minimum final_score for an authoritative chunk to be considered evidence.
RELEVANCE_THRESHOLD: float = 0.10

#: Whether to include all authoritative sibling chunks from qualifying documents.
ENABLE_SIBLING_BOOST: bool = True

#: Number of top candidates to retrieve from each index before fusion.
RETRIEVAL_TOP_K: int = 10

ABSTENTION_RESPONSE: str = (
    "I don't have enough authoritative information to answer that question confidently. "
    "I'd recommend speaking with our support team who can provide accurate assistance. "
    "Would you like me to connect you with a human agent?"
)

ORDER_ASK_ID_RESPONSE: str = (
    "I'd be happy to help with your order! "
    "Could you please provide your order ID? "
    "It's in the format ORD-XXXX (for example, ORD-1001)."
)


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """The structured response returned by Agent.handle_message.

    Attributes
    ----------
    answer:
        The final, validated answer text to return to the customer.
    citations:
        List of source citation dicts: [{"filename": ..., "heading": ...}].
        Empty for order-path or abstention responses.
    handoff:
        True when the agent recommends escalating to a human agent.
    trace:
        Full list of TraceEvent objects emitted during this turn (for
        observability and evaluation).
    """

    answer: str
    citations: list[dict[str, str]] = field(default_factory=list)
    handoff: bool = False
    trace: list[TraceEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge-base index (built once at Agent instantiation)
# ---------------------------------------------------------------------------


class _KnowledgeIndex:
    """Holds the dense, BM25, and chunk indexes for the knowledge base."""

    def __init__(self) -> None:
        _logger.info("Building knowledge-base indexes…")
        docs = parse_all()
        chunks = chunk_all_documents(docs)
        self.chunk_index = build_chunk_index(chunks)
        self.dense_index = DenseIndex.build(chunks)
        self.bm25_index = BM25Index.build(chunks)
        _logger.info(
            "Knowledge-base indexes ready",
            extra={"chunk_count": len(chunks)},
        )

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
        """Run hybrid retrieval and return fused + precedence-marked chunks."""
        dense_ranking = self.dense_index.search(query, top_k=top_k)
        bm25_ranking = self.bm25_index.search(query, top_k=top_k)
        fused = reciprocal_rank_fusion(
            dense_ranking=dense_ranking,
            bm25_ranking=bm25_ranking,
            chunk_index=self.chunk_index,
            top_k=top_k,
        )
        return mark_authoritative(fused)


# ---------------------------------------------------------------------------
# Trace-collection helper
# ---------------------------------------------------------------------------


class _TraceCollector:
    """Accumulate TraceEvents for the current turn."""

    def __init__(self, session_id: str, turn_id: int) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        self.events: list[TraceEvent] = []

    def emit(self, stage: str, payload: dict[str, Any]) -> None:  # noqa: ANN401
        event = TraceEvent(
            session_id=self.session_id,
            turn_id=self.turn_id,
            stage=stage,  # type: ignore[arg-type]
            payload=payload,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.events.append(event)
        emit_trace(event)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """The orchestration entrypoint.

    Parameters
    ----------
    knowledge_index:
        An optional pre-built ``_KnowledgeIndex``.  If not provided, the
        index is built from the knowledge base on instantiation.  Pass a
        pre-built index in tests to avoid rebuilding it per test.
    """

    def __init__(self, knowledge_index: _KnowledgeIndex | None = None) -> None:
        self._kb = knowledge_index or _KnowledgeIndex()
        self._conflict_detector = ConflictDetector()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def handle_message(self, session_id: str, text: str) -> AgentResponse:
        """Process a single user message and return an AgentResponse.

        Parameters
        ----------
        session_id:
            Opaque session identifier.  Sessions are isolated — state from
            one session_id never leaks to another.
        text:
            The user's raw message text.

        Returns
        -------
        AgentResponse
            Contains the final answer, citations, handoff flag, and full trace.
        """
        # Determine current turn index.
        turns = SESSION_STORE.get_turns(session_id)
        turn_id = len(turns)

        tracer = _TraceCollector(session_id, turn_id)

        # --- 1. Session context ------------------------------------------
        last_order_id = SESSION_STORE.get_last_order_id(session_id)
        last_topic = SESSION_STORE.get_last_topic(session_id)
        history = SESSION_STORE.build_history(session_id)

        # Append the current user message to history for the LLM call.
        history.append({"role": "user", "content": text})

        # --- 2. Deterministic routing ------------------------------------
        decision: RouterDecision = route(
            session_id=session_id,
            turn_id=turn_id,
            user_message=text,
            last_order_id=last_order_id,
        )

        # Emit the routing decision through the local tracer so the event
        # appears in AgentResponse.trace (route() also calls emit_trace()
        # for the global log, so we only need to add it here).
        tracer.emit(
            "router",
            {
                "path": decision.path,
                "order_id": decision.order_id,
                "reason": decision.reason,
                "message_snippet": text[:120],
            },
        )

        # --- 3. Branch ---------------------------------------------------
        if decision.path == "order_ask_id":
            return self._handle_order_ask_id(session_id, text, turn_id, tracer)

        if decision.path in ("order_direct", "order_followup"):
            return self._handle_order_path(
                session_id=session_id,
                text=text,
                turn_id=turn_id,
                order_id=decision.order_id,
                history=history,
                tracer=tracer,
            )

        # decision.path == "knowledge"
        return self._handle_knowledge_path(
            session_id=session_id,
            text=text,
            turn_id=turn_id,
            history=history,
            last_topic=last_topic,
            tracer=tracer,
        )

    # ------------------------------------------------------------------
    # Branch handlers
    # ------------------------------------------------------------------

    def _handle_order_ask_id(
        self,
        session_id: str,
        text: str,
        turn_id: int,
        tracer: _TraceCollector,
    ) -> AgentResponse:
        """Return a deterministic 'please provide your order ID' response."""
        tracer.emit("response", {"path": "order_ask_id", "deterministic": True})
        answer = validate_response(
            ORDER_ASK_ID_RESPONSE, [], session_id, turn_id
        )
        self._save_turn(session_id, text, answer, order_id=None, topic=None)
        return AgentResponse(
            answer=answer,
            citations=[],
            handoff=False,
            trace=tracer.events,
        )

    def _handle_order_path(
        self,
        session_id: str,
        text: str,
        turn_id: int,
        order_id: str | None,
        history: list[dict[str, str]],
        tracer: _TraceCollector,
    ) -> AgentResponse:
        """Call the order tool, build an evidence pack, and let the LLM phrase the answer."""
        tracer.emit("tool_call", {"function": "lookup_order", "order_id": order_id})

        result: SafeOrderResult = lookup_order(order_id)

        tracer.emit(
            "tool_result",
            {
                "order_id": result.order_id,
                "found": result.found,
                "status": result.status,
            },
        )

        evidence = format_evidence_pack(
            auth_chunks=[],
            conflict=None,
            order_result=result,
        )

        tracer.emit("llm_call", {"path": "order"})
        llm_answer, _ = llm_client.call_llm(
            rules=APP_RULES_TEXT,
            history=history,
            evidence_pack=evidence,
            include_order_tool=False,
        )

        answer = validate_response(llm_answer, [], session_id, turn_id)

        # Persist the established order_id in session for follow-up turns.
        established_order_id = result.order_id if result.found else order_id
        self._save_turn(session_id, text, answer, order_id=established_order_id, topic=None)

        tracer.emit("response", {"path": "order", "order_found": result.found})
        return AgentResponse(
            answer=answer,
            citations=[],
            handoff=False,
            trace=tracer.events,
        )

    def _handle_knowledge_path(
        self,
        session_id: str,
        text: str,
        turn_id: int,
        history: list[dict[str, str]],
        last_topic: str | None,
        tracer: _TraceCollector,
    ) -> AgentResponse:
        """Run retrieval → precedence → conflict → LLM for knowledge queries."""

        # --- Retrieval ---------------------------------------------------
        # Augment query with session topic for better context narrowing.
        query = text
        if last_topic:
            query = f"{last_topic}: {text}"

        chunks = self._kb.search(query)

        tracer.emit(
            "retrieval",
            {
                "query": query,
                "total_chunks": len(chunks),
                "authoritative_chunks": sum(1 for c in chunks if c.is_authoritative),
            },
        )

        # --- Precedence filter -------------------------------------------
        auth_chunks = [c for c in chunks if c.is_authoritative]

        tracer.emit(
            "precedence",
            {
                "auth_count": len(auth_chunks),
                "threshold": RELEVANCE_THRESHOLD,
                "above_threshold": sum(
                    1 for c in auth_chunks if c.final_score >= RELEVANCE_THRESHOLD
                ),
            },
        )

        # Filter to chunks above the relevance threshold, applying sibling boost if enabled.
        if ENABLE_SIBLING_BOOST:
            doc_scores: dict[str, float] = {}
            doc_max: dict[str, float] = {}
            for rc in auth_chunks:
                fn = rc.chunk.filename
                doc_scores[fn] = doc_scores.get(fn, 0.0) + rc.final_score
                if fn not in doc_max or rc.final_score > doc_max[fn]:
                    doc_max[fn] = rc.final_score

            qualifying_files = {fn for fn, m in doc_max.items() if m >= RELEVANCE_THRESHOLD}

            existing_by_id = {rc.chunk.chunk_id: rc for rc in auth_chunks}
            boosted_auth: list[RetrievedChunk] = []

            if hasattr(self._kb, "chunk_index") and self._kb.chunk_index:
                for chunk_id, chunk in self._kb.chunk_index.items():
                    if chunk.filename in qualifying_files:
                        is_auth = (
                            chunk.status == "active"
                            and chunk.policy_authority == "official"
                            and chunk.audience == "customer"
                        )
                        if is_auth:
                            if chunk_id in existing_by_id:
                                boosted_auth.append(existing_by_id[chunk_id])
                            else:
                                boosted_auth.append(
                                    RetrievedChunk(
                                        chunk=chunk,
                                        dense_score=0.0,
                                        bm25_score=0.0,
                                        rrf_score=0.0,
                                        final_score=doc_max.get(
                                            chunk.filename, RELEVANCE_THRESHOLD
                                        ),
                                        is_authoritative=True,
                                    )
                                )
            else:
                boosted_auth = [c for c in auth_chunks if c.chunk.filename in qualifying_files]

            boosted_auth.sort(
                key=lambda rc: (doc_scores.get(rc.chunk.filename, 0.0), rc.final_score),
                reverse=True,
            )
            relevant_auth = boosted_auth
        else:
            relevant_auth = [c for c in auth_chunks if c.final_score >= RELEVANCE_THRESHOLD]

        # --- Abstention gate ---------------------------------------------
        if not relevant_auth:
            tracer.emit(
                "response",
                {"path": "knowledge", "abstention": True, "reason": "no_relevant_auth_chunks"},
            )
            answer = validate_response(ABSTENTION_RESPONSE, [], session_id, turn_id)
            self._save_turn(session_id, text, answer, order_id=None, topic=last_topic)
            return AgentResponse(
                answer=answer,
                citations=[],
                handoff=True,
                trace=tracer.events,
            )

        # --- Conflict detection ------------------------------------------
        conflict: ConflictResult = self._conflict_detector.detect(chunks)

        tracer.emit(
            "conflict",
            {
                "has_conflict": conflict.has_conflict,
                "explanation": conflict.explanation,
                "conflicting_filenames": [c.filename for c in conflict.conflicting_chunks],
            },
        )

        # --- Evidence pack -----------------------------------------------
        evidence = format_evidence_pack(
            auth_chunks=relevant_auth,
            conflict=conflict,
            order_result=None,
        )

        # --- LLM call ----------------------------------------------------
        tracer.emit(
            "llm_call",
            {
                "path": "knowledge",
                "auth_chunk_count": len(relevant_auth),
                "conflict": conflict.has_conflict,
                "include_order_tool": True,
            },
        )

        llm_answer, fn_call_args = llm_client.call_llm(
            rules=APP_RULES_TEXT,
            history=history,
            evidence_pack=evidence,
            include_order_tool=True,
        )

        # --- Fallback: LLM emitted a lookup_order function call ----------
        order_result: SafeOrderResult | None = None
        if fn_call_args is not None:
            fallback_order_id = fn_call_args.get("order_id")
            tracer.emit(
                "tool_call",
                {"function": "lookup_order", "order_id": fallback_order_id, "via": "llm_fallback"},
            )
            order_result = lookup_order(fallback_order_id)
            tracer.emit(
                "tool_result",
                {"order_id": order_result.order_id, "found": order_result.found},
            )
            # Re-call LLM with the enriched evidence pack.
            evidence2 = format_evidence_pack(
                auth_chunks=relevant_auth,
                conflict=conflict,
                order_result=order_result,
            )
            tracer.emit("llm_call", {"path": "knowledge_with_order_fallback"})
            llm_answer, _ = llm_client.call_llm(
                rules=APP_RULES_TEXT,
                history=history,
                evidence_pack=evidence2,
                include_order_tool=False,
            )

        # --- Build citations from authoritative chunks -------------------
        citations = [
            {"filename": rc.chunk.filename, "heading": rc.chunk.heading_path}
            for rc in relevant_auth[:5]
        ]

        # --- Safety validation ------------------------------------------
        answer = validate_response(llm_answer, relevant_auth, session_id, turn_id)

        # --- Determine handoff ------------------------------------------
        # Conflicts always require human confirmation.
        handoff = conflict.has_conflict

        # Infer topic from top authoritative chunk (for session context).
        topic = relevant_auth[0].chunk.topic if relevant_auth else last_topic

        # Persist established order if fallback fired.
        established_order_id: str | None = None
        if order_result is not None and order_result.found:
            established_order_id = order_result.order_id

        self._save_turn(
            session_id, text, answer,
            order_id=established_order_id,
            topic=topic,
        )

        tracer.emit(
            "response",
            {
                "path": "knowledge",
                "abstention": False,
                "conflict": conflict.has_conflict,
                "handoff": handoff,
                "citation_count": len(citations),
            },
        )

        return AgentResponse(
            answer=answer,
            citations=citations,
            handoff=handoff,
            trace=tracer.events,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_turn(
        self,
        session_id: str,
        user_msg: str,
        agent_response: str,
        order_id: str | None,
        topic: str | None,
    ) -> None:
        """Persist the completed turn to the session store."""
        SESSION_STORE.add_turn(
            session_id,
            Turn(
                user_msg=user_msg,
                agent_response=agent_response,
                order_id=order_id,
                topic=topic,
            ),
        )
