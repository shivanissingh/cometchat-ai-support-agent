"""
app/config.py — Central configuration and logging bootstrap.

Security rule (for all future agents):
    NEVER interpolate secret values (API keys, tokens, passwords) into log
    messages or structured log payloads. Log only non-sensitive metadata
    such as model names, log levels, and boolean presence flags.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()  # reads .env in the current working directory (or any parent)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record, with a fixed set of fields.

    Security: do not include GEMINI_API_KEY or any other secret in the output.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any "extra" kwargs passed to the logging call.
        # Callers must NOT pass secret values in extra — see module docstring.
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "taskName",
            }:
                log_obj[key] = value
        return json.dumps(log_obj)


def _bootstrap_logging() -> None:
    """Configure the root logger to emit structured JSON lines."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))


_bootstrap_logging()
logger = logging.getLogger(__name__)

logger.info(
    "Config loaded",
    extra={
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "log_level": LOG_LEVEL,
        # NOTE: GEMINI_API_KEY is intentionally never logged — see module docstring.
        "api_key_present": bool(GEMINI_API_KEY),
    },
)


# ---------------------------------------------------------------------------
# Model resolution helper
# ---------------------------------------------------------------------------


def resolve_model(client) -> str:  # noqa: ANN001
    """Return the model name to use for the current request.

    TODO (orchestrator phase): Replace this stub with a live capability probe.
    The probe should query the Gemini client for available models, filter to
    those that support the required capabilities (e.g. function calling,
    long context), and fall back gracefully to GEMINI_MODEL if the probe
    fails or times out.  The resolved name should be cached per process
    startup to avoid repeated network calls.

    For now, the configured GEMINI_MODEL is returned unchanged.
    """
    return GEMINI_MODEL
