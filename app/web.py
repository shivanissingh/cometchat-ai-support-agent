"""
app/web.py — Streamlit chat UI for the CometChat RAG Support Agent.

UI Features
-----------
* Modern Header & Aesthetics:
  - Custom themed layout with responsive styling and branded headers.
  - Interactive quick-prompt suggestion pills for rapid exploration.
* Chat Experience:
  - User and assistant conversation history rendered with rich markdown.
  - Inline source citations displayed as clean expandable reference cards.
  - High-visibility escalation callout (🔺) when human handoff is recommended.
* Observability & Debug Panel:
  - Sidebar toggle to inspect the structured, scrubbed trace of the latest turn.
  - Stage-by-stage visual breakdown (Router ➜ Retrieval ➜ Precedence ➜ Conflict ➜ LLM ➜ Response).
  - Raw JSON viewer with syntax styling for auditability.
* Session Isolation:
  - Fully scoped to `st.session_state` (independent per tab / browser session).

Run with::

    streamlit run app/web.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is in sys.path (supports `streamlit run app/web.py`)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CometChat Support Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for polished, modern look & feel
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 18px;
    }
    .main-header-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header-sub {
        font-size: 0.9rem;
        color: #888;
        margin-top: -4px;
    }
    .badge-online {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    /* Citation pills */
    .citation-card {
        background: rgba(128, 128, 128, 0.08);
        border-left: 3px solid #0066cc;
        border-radius: 4px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    /* Handoff notice */
    .handoff-box {
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #d39e00;
    }
    /* Quick prompt button styling */
    .stButton button {
        border-radius: 20px;
        font-size: 0.85rem;
        padding: 4px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Agent singleton — built once per Streamlit server process.
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
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


_init_session()


# ---------------------------------------------------------------------------
# Sidebar: Session info, Controls, and Debug Trace Inspector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💬 CometChat Support")
    st.markdown(
        f"<span class='badge-online'>ACTIVE</span> &nbsp; "
        f"<small>Session: <code>{st.session_state.session_id[:8]}...</code></small>",
        unsafe_allow_html=True,
    )
    st.caption("AI-Powered Customer Support with Precedence RAG")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_trace = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.divider()

    show_debug = st.toggle("🔍 Show debug trace", value=False)

    if show_debug:
        if st.session_state.last_trace:
            st.markdown("#### 🔬 Latest Turn Trace")

            # Extract high-level summary from events
            stages_present = [e.stage for e in st.session_state.last_trace]
            router_event = next(
                (e for e in st.session_state.last_trace if e.stage == "router"),
                None,
            )
            response_event = next(
                (e for e in st.session_state.last_trace if e.stage == "response"),
                None,
            )

            path_name = (
                router_event.payload.get("path", "unknown")
                if router_event
                else "N/A"
            )
            is_handoff = (
                response_event.payload.get("handoff", False)
                if response_event
                else False
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Route Path", path_name)
            with col2:
                st.metric("Handoff", "Yes 🔺" if is_handoff else "No ✔")

            st.caption(f"Stages Executed: `{' ➜ '.join(stages_present)}`")

            with st.expander("Detailed Stage Payloads", expanded=True):
                for event in st.session_state.last_trace:
                    stage_icon = {
                        "router": "🎯",
                        "retrieval": "🔎",
                        "precedence": "⚖️",
                        "conflict": "🛡️",
                        "tool_call": "⚙️",
                        "tool_result": "📥",
                        "llm_call": "🤖",
                        "validation": "🔒",
                        "response": "💬",
                    }.get(event.stage, "📌")

                    st.markdown(f"**{stage_icon} Stage: `{event.stage}`**")
                    st.json(event.payload)
        else:
            st.info("No trace events recorded yet. Send a message to inspect.")

    st.divider()
    with st.expander("ℹ️ Sample Queries", expanded=False):
        st.markdown(
            """
            - **Policy:** *What is your return policy?*
            - **Order Lookup:** *Where is ORD-1007?*
            - **Damaged Item:** *My tumbler arrived broken.*
            - **Canadian Orders:** *Can I return from Canada?*
            - **Final Sale:** *Is Ridge Daypack returnable?*
            """
        )


# ---------------------------------------------------------------------------
# Main Chat Area: Header & Conversation
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class='main-header'>
        <div>
            <h2 class='main-header-title'>Aster & Row Support Assistant</h2>
            <div class='main-header-sub'>Powered by CometChat AI Precedence RAG Orchestrator</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# If chat is empty, show welcoming prompt pills
if not st.session_state.messages:
    st.markdown("👋 **Welcome! How can I assist you with your orders or policies today?**")
    st.caption("Try one of these quick queries:")
    pills = [
        "📦 Where is ORD-1007?",
        "🔄 What is your return policy?",
        "🎒 Can I return a final-sale item?",
        "❓ What is the flibbertigibbet quotient?",
    ]
    pcol1, pcol2 = st.columns(2)
    for i, pill in enumerate(pills):
        target_col = pcol1 if i % 2 == 0 else pcol2
        with target_col:
            if st.button(pill, key=f"pill_{i}", use_container_width=True):
                # Strip leading emoji for clean message
                clean_text = pill.split(" ", 1)[1] if " " in pill else pill
                st.session_state.pending_prompt = clean_text
                st.rerun()

# Render existing messages
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # Handoff alert badge
            if msg.get("handoff"):
                st.warning(
                    "🔺 **Recommend contacting human support** — "
                    "This inquiry requires human assistance.",
                    icon="🔺",
                )

            # Citations list
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
                    for cite in citations:
                        filename = cite.get("filename", "Unknown")
                        heading = cite.get("heading", "")
                        card_html = (
                            f"<div class='citation-card'>📄 <b>{filename}</b>"
                            f" — <i>{heading}</i></div>"
                        )
                        st.markdown(card_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat Input & Processing
# ---------------------------------------------------------------------------
prompt = st.chat_input("Type your question here…")

# Check if a prompt came from quick pills or the chat input bar
active_prompt = prompt or st.session_state.pending_prompt
if active_prompt:
    st.session_state.pending_prompt = None

    # Render User Message immediately
    st.session_state.messages.append(
        {"role": "user", "content": active_prompt, "citations": [], "handoff": False, "trace": []}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_prompt)

    # Process through orchestrator Agent
    try:
        agent = _get_agent()
        with st.spinner("Analyzing knowledge base & generating response…"):
            response = agent.handle_message(
                session_id=st.session_state.session_id,
                text=active_prompt,
            )

        # Store trace for debug panel
        st.session_state.last_trace = response.trace

        # Record assistant message
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

        # Render Assistant Response
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response.answer)

            if response.handoff:
                st.warning(
                    "🔺 **Recommend contacting human support** — "
                    "This inquiry requires human assistance.",
                    icon="🔺",
                )

            if response.citations:
                with st.expander(f"📚 Sources ({len(response.citations)})", expanded=False):
                    for cite in response.citations:
                        filename = cite.get("filename", "Unknown")
                        heading = cite.get("heading", "")
                        card_html = (
                            f"<div class='citation-card'>📄 <b>{filename}</b>"
                            f" — <i>{heading}</i></div>"
                        )
                        st.markdown(card_html, unsafe_allow_html=True)

        # Rerun to update sidebar debug panel seamlessly
        st.rerun()

    except Exception as err:
        st.error(f"⚠️ An error occurred while processing your request: {err}")
