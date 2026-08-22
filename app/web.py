"""
app/web.py — Streamlit chat UI for the CometChat RAG Support Agent.

UI structure
------------
* Left sidebar:
  - App title and session info.
  - "Show debug trace" toggle — when enabled the most recent turn's scrubbed
    trace is shown in an expandable section below the sidebar controls.

* Main area:
  - Chat-style conversation view rendered with ``st.chat_message``.
  - User messages are displayed as "user" bubbles.
  - Assistant messages are displayed as "assistant" bubbles with:
      • The answer text.
      • A 🔺 handoff callout (``st.warning``) when ``handoff=True``.
      • An expandable "Sources" section listing citations when present.

Session isolation
-----------------
``st.session_state`` is used for ALL mutable state so that each browser tab /
user gets a completely independent session.  No module-level globals carry
turn data — only the Agent instance is shared (it is stateless apart from
the knowledge index, which is read-only after build).

Security
--------
The debug panel shows ``AgentResponse.trace`` directly.  This list contains
``TraceEvent`` objects whose payloads have already been scrubbed by the
observability layer (``scrub_payload`` in ``app.observability.tracer``).
The Streamlit layer never touches raw internal order state.

Run with::

    streamlit run app/web.py
"""

from __future__ import annotations

import json
import uuid

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CometChat Support Agent",
    page_icon="💬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Agent singleton — built once per Streamlit server process.
# The knowledge index is expensive to build; share it across sessions.
# Agent itself carries no per-user state beyond the shared knowledge index.
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_agent():  # type: ignore[return]
    """Build and cache the Agent (index construction happens once)."""
    from app.agent.orchestrator import Agent

    return Agent()


# ---------------------------------------------------------------------------
# Per-session state initialisation
# ---------------------------------------------------------------------------


def _init_session() -> None:
    """Initialise session_state keys on the very first run for this tab."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        #              "citations": list, "handoff": bool, "trace": list}
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = []


_init_session()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💬 CometChat Support")
    st.caption(f"Session: `{st.session_state.session_id}`")
    st.divider()

    show_debug = st.toggle("Show debug trace", value=False)

    if show_debug and st.session_state.last_trace:
        st.subheader("🔍 Debug trace (most recent turn)")
        with st.expander("Trace events", expanded=True):
            for event in st.session_state.last_trace:
                st.json(
                    {
                        "stage": event.stage,
                        "turn_id": event.turn_id,
                        "timestamp": event.timestamp,
                        "payload": event.payload,
                    }
                )
    elif show_debug:
        st.info("No trace yet — ask a question first.")

# ---------------------------------------------------------------------------
# Main chat area — render conversation history
# ---------------------------------------------------------------------------

st.header("CometChat AI Support Agent")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Handoff badge
            if msg.get("handoff"):
                st.warning(
                    "🔺 **Recommend contacting human support**",
                    icon="🔺",
                )

            # Citations
            citations = msg.get("citations", [])
            if citations:
                with st.expander("📚 Sources", expanded=False):
                    for cite in citations:
                        filename = cite.get("filename", "")
                        heading = cite.get("heading", "")
                        st.markdown(f"- **{filename}** — {heading}")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question…"):
    # Display user message immediately
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "citations": [], "handoff": False, "trace": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the agent
    agent = _get_agent()
    with st.spinner("Thinking…"):
        response = agent.handle_message(
            session_id=st.session_state.session_id,
            text=prompt,
        )

    # Store trace for sidebar debug panel
    st.session_state.last_trace = response.trace

    # Build assistant message record
    assistant_msg = {
        "role": "assistant",
        "content": response.answer,
        "citations": response.citations,
        "handoff": response.handoff,
        "trace": [
            json.loads(event.model_dump_json())
            for event in response.trace
        ],
    }
    st.session_state.messages.append(assistant_msg)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response.answer)

        if response.handoff:
            st.warning(
                "🔺 **Recommend contacting human support**",
                icon="🔺",
            )

        if response.citations:
            with st.expander("📚 Sources", expanded=False):
                for cite in response.citations:
                    filename = cite.get("filename", "")
                    heading = cite.get("heading", "")
                    st.markdown(f"- **{filename}** — {heading}")

    # Rerun to refresh sidebar debug panel immediately after the turn
    st.rerun()
