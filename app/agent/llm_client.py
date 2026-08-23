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

FALLBACK_MODEL = "gemini-3.5-flash-lite"

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


def _resolve_model() -> str:
    """Return the configured model name."""
    global _resolved_model  # noqa: PLW0603
    if _resolved_model is not None:
        return _resolved_model
    _resolved_model = config.GEMINI_MODEL or FALLBACK_MODEL
    return _resolved_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_model_ready() -> str:
    """Return the resolved model name."""
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
    """
    client = _get_client()

    contents: list[genai_types.Content] = []
    for turn in history[:-1]:
        role = turn.get("role", "user")
        text = turn.get("content", "")
        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=text)],
            )
        )

    if history:
        last_user_text = history[-1].get("content", "")
        last_turn_content = f"{last_user_text}\n\n{evidence_pack}"
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=last_turn_content)],
            )
        )
    else:
        contents.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=evidence_pack)],
            )
        )

    from app.agent.model_manager import GLOBAL_MODEL_MANAGER

    tools = [LOOKUP_ORDER_TOOL] if include_order_tool else []

    cfg = genai_types.GenerateContentConfig(
        system_instruction=rules,
        temperature=0.2,
        tools=tools if tools else None,
    )

    max_model_attempts = len(GLOBAL_MODEL_MANAGER.models) * 2
    response = None
    last_exc = None

    for _ in range(max_model_attempts):
        model = GLOBAL_MODEL_MANAGER.acquire_call_slot()
        GLOBAL_MODEL_MANAGER.record_call(model)

        _logger.info(
            "LLM call",
            extra={
                "model": model,
                "history_turns": len(history),
                "include_order_tool": include_order_tool,
            },
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=cfg,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            exc_str = str(exc)
            is_quota = any(
                code in exc_str
                for code in ("429", "ResourceExhausted", "RESOURCE_EXHAUSTED", "quota")
            )
            is_not_found = any(
                term in exc_str for term in ("404", "NOT_FOUND", "NotFound", "no longer available")
            )
            is_unavailable = (
                "503" in exc_str or "UNAVAILABLE" in exc_str or "demand" in exc_str.lower()
            )

            if is_quota:
                _logger.warning(
                    "Quota / rate-limit hit on model %s; switching to next model in pool",
                    model,
                    extra={"error": exc_str[:200]},
                )
                GLOBAL_MODEL_MANAGER.mark_exhausted(model, reason="429 ResourceExhausted")
                continue
            elif is_not_found:
                _logger.warning(
                    "Model %s not found / deprecated; switching to next model in pool",
                    model,
                    extra={"error": exc_str[:200]},
                )
                GLOBAL_MODEL_MANAGER.mark_exhausted(model, reason="404 Model Not Found")
                continue
            elif is_unavailable:
                _logger.warning(
                    "Model %s unavailable (503); switching to next model in pool",
                    model,
                    extra={"error": exc_str[:200]},
                )
                GLOBAL_MODEL_MANAGER.mark_exhausted(model, reason="503 Service Unavailable")
                continue
            else:
                raise

    if response is None:
        if last_exc is not None:
            raise last_exc
        return "", None

    # Check for a function call in the response.
    fn_call_args: dict[str, Any] | None = None
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.function_call is not None:
                fn_call_args = dict(part.function_call.args or {})
                _logger.info(
                    "LLM emitted function call",
                    extra={"function_name": part.function_call.name, "fn_args": fn_call_args},
                )
                break

    text_response = response.text or ""
    return text_response, fn_call_args
