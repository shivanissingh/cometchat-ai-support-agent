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
from collections import deque
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

# ANSI Color codes for beautiful terminal output
_GREEN = "[92m"
_RED = "[91m"
_YELLOW = "[93m"
_BLUE = "[94m"
_CYAN = "[96m"
_BOLD = "[1m"
_RESET = "[0m"

# The 6 canonical model options in preference order
DEFAULT_MODELS: list[str] = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
]

# Safety limits
MAX_RPM: int = 3  # Strict RPM cap: maximum 3 calls per minute
MIN_CALL_GAP_SECONDS: float = 20.0  # Proper time gap between consecutive calls (60s / 3 = 20s)
FLASH_RPD_CAP: int = 18  # Flash models daily cap (max 18 calls/day)
LITE_RPD_CAP: int = 490  # Lite models daily cap (max 490 calls/day)
WINDOW_SECONDS: float = 60.0


def get_rpd_cap(model: str) -> int:
    """Return the daily safety cap for a given model name."""
    if "lite" in model.lower():
        return LITE_RPD_CAP
    return FLASH_RPD_CAP


@dataclass
class ModelStats:
    model: str
    daily_count: int = 0
    call_timestamps: deque[float] = field(default_factory=deque)
    quota_exhausted: bool = False

    @property
    def rpd_cap(self) -> int:
        return get_rpd_cap(self.model)

    def prune_window(self, now: float) -> None:
        """Remove timestamps older than WINDOW_SECONDS."""
        while self.call_timestamps and (now - self.call_timestamps[0] >= WINDOW_SECONDS):
            self.call_timestamps.popleft()

    @property
    def rpm_count(self) -> int:
        now = time.time()
        self.prune_window(now)
        return len(self.call_timestamps)

    def is_rpd_available(self) -> bool:
        """True if model has not exhausted its daily limit cap."""
        return not self.quota_exhausted and self.daily_count < self.rpd_cap

    def calculate_rpm_wait(self, now: float) -> float:
        """Calculate required sleep time to strictly enforce RPM <= 3 and proper time gaps."""
        self.prune_window(now)
        wait = 0.0

        # Gap check: ensure at least MIN_CALL_GAP_SECONDS since the previous call
        if self.call_timestamps:
            last_call = self.call_timestamps[-1]
            gap_needed = (last_call + MIN_CALL_GAP_SECONDS) - now
            if gap_needed > wait:
                wait = gap_needed

        # Sliding window check: ensure < MAX_RPM calls in the last 60s
        if len(self.call_timestamps) >= MAX_RPM:
            oldest = self.call_timestamps[0]
            window_needed = (oldest + WINDOW_SECONDS) - now + 0.5
            if window_needed > wait:
                wait = window_needed

        return max(0.0, wait)


class ModelManager:
    """Thread-safe manager enforcing RPM timer spacing and RPD-only model switching."""

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

    def mark_exhausted(self, model: str, reason: str = "429 Quota Exhausted") -> None:
        """Mark a model as having exhausted its daily quota and advance to next model."""
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
            f"║    Active: {_GREEN}{_BOLD}{new_model:<16}{_RESET}"
            f"{_YELLOW} Quota: {quota_str:<19}║"
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
        """Select the active model and sleep the required gap to strictly enforce RPM <= 3.

        Model is NOT switched for RPM. It only sleeps the exact required timer gap.
        """
        model = self.get_current_model()
        st = self.stats[model]

        if not st.is_rpd_available():
            old = model
            self._advance_to_next_available()
            model = self.get_current_model()
            self._print_rotation_event(old_model=old, reason=f"Daily Cap ({st.rpd_cap}) Reached")
            st = self.stats[model]

        now = time.time()
        wait = st.calculate_rpm_wait(now)

        if wait > 0:
            msg = f"  {_CYAN}⏱ [Rate Pacing] Waiting {wait:.1f}s (<= 3 RPM) on {model}...{_RESET}\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            time.sleep(wait)

        return model

    def record_call(self, model: str) -> None:
        """Record an initiated call timestamp and increment daily count for the model."""
        if model not in self.stats:
            self.stats[model] = ModelStats(model=model)
        now = time.time()
        st = self.stats[model]
        st.daily_count += 1
        st.call_timestamps.append(now)
        st.prune_window(now)

        _logger.info(
            "Model call recorded",
            extra={
                "model": model,
                "daily_calls": st.daily_count,
                "daily_cap": st.rpd_cap,
                "rpm_in_window": len(st.call_timestamps),
                "rpm_cap": MAX_RPM,
            },
        )

        if st.daily_count >= st.rpd_cap:
            old = model
            self._advance_to_next_available()
            self._print_rotation_event(old_model=old, reason=f"Daily Cap ({st.rpd_cap}) Reached")


GLOBAL_MODEL_MANAGER = ModelManager()
