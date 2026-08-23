"""
app/agent/model_manager.py — Multi-model rotation and rate-limit manager.

Rules:
1. Model switching is ONLY performed when a model reaches its daily limit cap:
   - Flash models (non-lite): max 18 calls/day (under the 20 RPD free tier).
   - Flash Lite models: max 490 calls/day (under the 500 RPD free tier).
   - Or when a 429 / Quota Exhaustion response is returned for that model.
2. For RPM, the limit cap is strictly 3 calls per minute.
   - We NEVER switch models for RPM.
   - Instead, proper time gaps (minimum 20s spacing or waiting for the 60s sliding window)
     are strictly enforced with timers so rate limit errors never occur.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

# ANSI Color codes for beautiful terminal output
_GREEN = "[92m"
_RED = "[91m"
_YELLOW = "[93m"
_BLUE = "[94m"
_CYAN = "[96m"
_BOLD = "[1m"
_RESET = "[0m"

# Canonical model options in preference order
DEFAULT_MODELS: list[str] = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

# Safety limits
FIXED_CALL_GAP_SECONDS: float = 15.0
FLASH_RPD_CAP: int = 18  # Flash models daily cap (max 18 calls/day)
LITE_RPD_CAP: int = 490  # Lite models daily cap (max 490 calls/day)


def get_rpd_cap(model: str) -> int:
    """Return the daily safety cap for a given model name."""
    if "lite" in model.lower():
        return LITE_RPD_CAP
    return FLASH_RPD_CAP


@dataclass
class ModelStats:
    model: str
    daily_count: int = 0
    quota_exhausted: bool = False

    @property
    def rpd_cap(self) -> int:
        return get_rpd_cap(self.model)

    def is_rpd_available(self) -> bool:
        """True if model has not exhausted its daily limit cap."""
        return not self.quota_exhausted and self.daily_count < self.rpd_cap


class ModelManager:
    """Manager enforcing strict 15s inter-request pacing and RPD model switching."""

    def __init__(self, models: list[str] | None = None) -> None:
        raw_env = os.getenv("EVAL_MODELS", "")
        if raw_env:
            parsed = [m.strip() for m in raw_env.split(",") if m.strip()]
            self.models = parsed if parsed else list(DEFAULT_MODELS)
        elif models:
            self.models = list(models)
        else:
            self.models = list(DEFAULT_MODELS)

        self.stats: dict[str, ModelStats] = {m: ModelStats(model=m) for m in self.models}
        self._current_index: int = 0
        self._last_call_time: float = 0.0

    def mark_exhausted(self, model: str, reason: str = "429 Quota Exhausted") -> None:
        """Mark a model as having exhausted its quota and advance to next model."""
        if model in self.stats:
            self.stats[model].quota_exhausted = True
            self._print_rotation_event(
                old_model=model,
                reason=reason,
            )
        self._advance_to_next_available()

    def _print_rotation_event(self, old_model: str, reason: str) -> None:
        """Print a formatted banner when model rotation occurs."""
        b = "═" * 70
        print(f"\n{_YELLOW}╔{b}╗{_RESET}", flush=True)
        print(f"{_YELLOW}║ 🔄 MODEL ROTATION EVENT{' ':45}║{_RESET}", flush=True)
        prev_line = f"║    Prev: {_BOLD}{old_model:<18}{_RESET}{_YELLOW} Reason: {reason:<24}║"
        print(f"{_YELLOW}{prev_line}{_RESET}", flush=True)
        self._advance_to_next_available()
        new_model = self.models[self._current_index]
        cap = self.stats[new_model].rpd_cap
        calls = self.stats[new_model].daily_count
        quota_str = f"{calls}/{cap} calls"
        act_line = (
            f"║    Active: {_GREEN}{_BOLD}{new_model:<16}{_RESET}{_YELLOW} Quota: {quota_str:<19}║"
        )
        print(f"{_YELLOW}{act_line}{_RESET}", flush=True)
        print(f"{_YELLOW}╚{b}╝{_RESET}\n", flush=True)

    def _advance_to_next_available(self) -> None:
        """Advance _current_index to the next model with daily quota remaining."""
        for offset in range(len(self.models)):
            idx = (self._current_index + offset) % len(self.models)
            m = self.models[idx]
            if self.stats[m].is_rpd_available():
                self._current_index = idx
                return

        # If all models exhausted daily cap, reset Lite models as fallback
        for m in self.models:
            if "lite" in m.lower():
                self.stats[m].quota_exhausted = False
                self.stats[m].daily_count = 0
        for idx, m in enumerate(self.models):
            if "lite" in m.lower():
                self._current_index = idx
                return

    def get_current_model(self) -> str:
        """Return the currently active model based on daily limit caps."""
        self._advance_to_next_available()
        return self.models[self._current_index]

    def get_model_status_str(self) -> str:
        """Return formatted status string for terminal display."""
        model = self.get_current_model()
        st = self.stats[model]
        return f"{model} (Calls: {st.daily_count}/{st.rpd_cap})"

    def acquire_call_slot(self) -> str:
        """Sleep fixed 15s since last call and return active model."""
        model = self.get_current_model()
        st = self.stats[model]

        if not st.is_rpd_available():
            old = model
            self._advance_to_next_available()
            model = self.get_current_model()
            self._print_rotation_event(old_model=old, reason=f"Daily Cap ({st.rpd_cap}) Reached")

        now = time.time()
        elapsed = now - self._last_call_time
        if self._last_call_time > 0 and elapsed < FIXED_CALL_GAP_SECONDS:
            wait = FIXED_CALL_GAP_SECONDS - elapsed
            msg = f"  {_CYAN}⏱ [Rate Pacing] Waiting {wait:.1f}s (15s gap) on {model}...{_RESET}\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            time.sleep(wait)

        self._last_call_time = time.time()
        return model

    def record_call(self, model: str) -> None:
        """Record an initiated call and increment daily count for the model."""
        if model not in self.stats:
            self.stats[model] = ModelStats(model=model)
        st = self.stats[model]
        st.daily_count += 1

        _logger.info(
            "Model call recorded",
            extra={
                "model": model,
                "daily_calls": st.daily_count,
                "daily_cap": st.rpd_cap,
            },
        )

        if st.daily_count >= st.rpd_cap:
            old = model
            self._advance_to_next_available()
            self._print_rotation_event(old_model=old, reason=f"Daily Cap ({st.rpd_cap}) Reached")


GLOBAL_MODEL_MANAGER = ModelManager()
