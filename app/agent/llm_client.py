"""
app/agent/llm_client.py — google-genai SDK wrapper with capability probe.

Capability probe
----------------
On the first call to ``call_llm`` (or an explicit ``ensure_model_ready()``),
a minimal test request is sent to the configured model with a trivial
function-calling schema.  If the request fails for any reason (auth error,
model not found, function calling unsupported), a structured warning is
logged and the resolved model falls back to ``gemini-2.5-flash`` for the
rest of the process lifetime.  The resolved model name is cached in the
module-level ``_resolved_model`` variable so the probe fires only once.

Security rule
-------------
NEVER log GEMINI_API_KEY.  Log only the model name and a boolean presence
flag for the key.

google-genai SDK note
---------------------
The ``google-genai`` package exposes ``google.genai`` and uses
``genai.Client(api_key=...)`` as the entry point.
"""

from __future__ import annotations

import logging
from typing import Any

import google.genai as genai
import google.genai.types as genai_types

from app import config

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_client: genai.Client | None = None
_resolved_model: str | None = None  # None = probe not yet run

FALLBACK_MODEL = "gemini-3.6-flash"

# ---------------------------------------------------------------------------
# lookup_order function schema for Gemini function calling
# ---------------------------------------------------------------------------

LOOKUP_ORDER_TOOL = genai_types.Tool(
    function_declarations=[
        genai_types.FunctionDeclaration(
            name="lookup_order",
            description=(
                "Look up an order by its order ID and return safe, "
                "customer-facing order information."
            ),
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "order_id": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="Order ID in the format ORD-XXXX (e.g. ORD-1001).",
                    )
                },
                required=["order_id"],
            ),
        )
    ]
)

# Minimal trivial tool used only during the capability probe.
_PROBE_TOOL = genai_types.Tool(
    function_declarations=[
        genai_types.FunctionDeclaration(
            name="probe_fn",
            description="Capability probe — always ignore.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "x": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="probe parameter",
                    )
                },
                required=[],
            ),
        )
    ]
)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _get_client() -> genai.Client:
    """Return the singleton genai Client, creating it on first call."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _run_probe(client: genai.Client, model: str) -> bool:
    """Send a minimal test request; return True if model supports function calling."""
    try:
        response = client.models.generate_content(
            model=model,
            contents="Say 'ok'.",
            config=genai_types.GenerateContentConfig(
                tools=[_PROBE_TOOL],
                temperature=0,
            ),
        )
        # Any non-exception response counts as a pass — the model responded.
        _ = response.text  # access text to trigger any lazy errors
        return True
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Capability probe failed",
            extra={
                "probe_model": model,
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
            },
        )
        return False


def _resolve_model() -> str:
    """Run the capability probe (once) and return the resolved model name."""
    global _resolved_model  # noqa: PLW0603
    if _resolved_model is not None:
        return _resolved_model

    configured = config.GEMINI_MODEL
    client = _get_client()

    _logger.info(
        "Running capability probe",
        extra={"probe_model": configured, "api_key_present": bool(config.GEMINI_API_KEY)},
    )

    if _run_probe(client, configured):
        _resolved_model = configured
        _logger.info("Capability probe passed", extra={"resolved_model": _resolved_model})
    else:
        _resolved_model = FALLBACK_MODEL
        _logger.warning(
            "Capability probe failed — falling back",
            extra={"configured_model": configured, "fallback_model": _resolved_model},
        )

    return _resolved_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_model_ready() -> str:
    """Run the capability probe if not yet done; return the resolved model name.

    Useful for pre-warming at startup rather than waiting for the first
    ``call_llm`` invocation.
    """
    return _resolve_model()


def call_llm(
    rules: str,
    history: list[dict[str, str]],
    evidence_pack: str,
    include_order_tool: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    """Call the Gemini model and return the response text and any function call.

    Parameters
    ----------
    rules:
        The static application rules text (injected as system instructions).
    history:
        Recent conversation turns as a list of dicts with keys "role" and
        "content".  Role is either "user" or "model".
    evidence_pack:
        The pre-formatted evidence pack string (output of
        ``app.agent.prompts.format_evidence_pack``).  This is appended to
        the final user turn so it is clearly DATA, not system instructions.
    include_order_tool:
        If True, include the ``lookup_order`` function declaration so the
        model can emit a function call as a fallback when it believes an
        order lookup is required but the deterministic router did not catch it.

    Returns
    -------
    tuple[str, dict | None]
        (response_text, function_call_args_or_None)
        ``function_call_args`` is a dict like ``{"order_id": "ORD-1001"}``
        when the model emits a ``lookup_order`` function call; otherwise None.
    """
    model = _resolve_model()
    client = _get_client()

    # Build the contents list for the Gemini API.
    contents: list[genai_types.Content] = []

    # Inject prior history turns (user / model alternating).
    for turn in history:
        role = turn.get("role", "user")
        text = turn.get("content", "")
        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=text)],
            )
        )

    # Append the evidence pack to the last user content if history is non-empty,
    # otherwise create a standalone user content with just the evidence pack.
    if contents and contents[-1].role == "user":
        last_parts = list(contents[-1].parts or [])
        last_parts.append(genai_types.Part(text=f"\n\n{evidence_pack}"))
        contents[-1] = genai_types.Content(role="user", parts=last_parts)
    else:
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=evidence_pack)],
            )
        )

    tools = [LOOKUP_ORDER_TOOL] if include_order_tool else []

    cfg = genai_types.GenerateContentConfig(
        system_instruction=rules,
        temperature=0.2,
        tools=tools if tools else None,
    )

    _logger.info(
        "LLM call",
        extra={
            "model": model,
            "history_turns": len(history),
            "include_order_tool": include_order_tool,
        },
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=cfg,
    )

    # Check for a function call in the response.
    fn_call_args: dict[str, Any] | None = None
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.function_call is not None:
                fn_call_args = dict(part.function_call.args or {})
                _logger.info(
                    "LLM emitted function call",
                    extra={"function_name": part.function_call.name, "args": fn_call_args},
                )
                break

    text_response = response.text or ""
    return text_response, fn_call_args
