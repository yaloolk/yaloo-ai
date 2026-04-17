"""
API Fallback Management for Gemini API Keys.

Supports a 3-tier fallback chain: primary → secondary → tertiary.
Thread-safe for concurrent request handling.

To add a 4th key in future:
  1. Add QUATERNARY_GEMINI_API_KEY to settings.
  2. Add a "quaternary" entry in APIConfig.__init__() following the same pattern.
  3. No other changes needed — the retry logic is tier-count agnostic.
"""
import asyncio
import inspect
import logging
import threading
import time
from typing import Any, Callable, Dict
from functools import wraps

from app.core.config import Settings


logger = logging.getLogger(__name__)

# Ordered fallback chain — index 0 is tried first.
_TIER_NAMES = ("primary", "secondary", "tertiary")


class APIConfig:
    """Text API configuration with N-tier fallback support. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current_index: int = 0  # index into _TIER_NAMES

        self._tiers: Dict[str, Dict[str, Any]] = {
            "primary": {
                "api_key":     Settings.PRIMARY_GEMINI_API_KEY.get_secret_value(),
                "name":        "Primary Gemini API",
                "retry_count": 0,
                "max_retries": 3,
            },
            "secondary": {
                "api_key":     Settings.SECONDARY_GEMINI_API_KEY.get_secret_value(),
                "name":        "Secondary Gemini API",
                "retry_count": 0,
                "max_retries": 3,
            },
            "tertiary": {
                "api_key":     Settings.TERTIARY_GEMINI_API_KEY.get_secret_value(),
                "name":        "Tertiary Gemini API",
                "retry_count": 0,
                "max_retries": 3,
            },
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _current_tier(self) -> str:
        return _TIER_NAMES[self._current_index]

    @property
    def _current_cfg(self) -> Dict[str, Any]:
        return self._tiers[self._current_tier]

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current_api_key(self) -> str:
        """Return the active API key."""
        with self._lock:
            return self._current_cfg["api_key"]

    def get_status(self) -> Dict[str, Any]:
        """Return a safe status snapshot (key masked)."""
        with self._lock:
            key = self._current_cfg["api_key"]
            return {
                "current_tier":    self._current_tier,
                "api_name":        self._current_cfg["name"],
                "current_key":     key[:10] + "...",
                "retry_count":     self._current_cfg["retry_count"],
                "max_retries":     self._current_cfg["max_retries"],
                "tiers_available": list(_TIER_NAMES),
            }

    def reset_to_primary(self) -> None:
        """Reset the active tier to primary and clear all retry counters."""
        with self._lock:
            if self._current_index != 0:
                self._current_index = 0
                for cfg in self._tiers.values():
                    cfg["retry_count"] = 0
                logger.info("Reset to primary Gemini API")

    def handle_api_error(self, error: Exception) -> bool:
        """
        Record a failure on the current tier and advance if exhausted.

        Returns:
            True  — caller should retry (same tier has attempts left, or moved to next tier).
            False — all tiers exhausted, caller should stop and raise.
        """
        with self._lock:
            cfg = self._current_cfg
            cfg["retry_count"] += 1
            logger.error(
                "API error on %s (attempt %d/%d): %s",
                cfg["name"], cfg["retry_count"], cfg["max_retries"], error,
            )

            if cfg["retry_count"] < cfg["max_retries"]:
                return True  # Still have retries on this tier.

            # Tier exhausted — try to advance.
            next_index = self._current_index + 1
            if next_index < len(_TIER_NAMES):
                self._current_index = next_index
                logger.warning(
                    "Switching to %s after exhausting %s",
                    self._current_cfg["name"], cfg["name"],
                )
                return True

            logger.error("All Gemini API tiers exhausted.")
            return False


# Singleton used across the application.
api_config = APIConfig()


# ── Retry decorator ───────────────────────────────────────────────────────────

def api_retry_with_fallback(max_attempts: int = 9):
    """
    Decorator that retries a function across the fallback tier chain.

    Works on both sync and async functions.
    Automatically injects the current api_key kwarg when the caller passes it.

    max_attempts defaults to 9 (3 tiers x 3 retries each).
    Raise this value if you add more tiers or increase max_retries.
    """
    def decorator(func: Callable) -> Callable:

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_error: Exception | None = None
                for attempt in range(max_attempts):
                    try:
                        if "api_key" in kwargs:
                            kwargs["api_key"] = api_config.get_current_api_key()
                        result = await func(*args, **kwargs)
                        api_config.reset_to_primary()
                        return result
                    except Exception as error:
                        last_error = error
                        logger.error("Async attempt %d/%d failed: %s", attempt + 1, max_attempts, error)
                        if not api_config.handle_api_error(error):
                            break
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(min(2 ** attempt, 10))
                logger.error("All async attempts failed. Last error: %s", last_error)
                raise last_error
            return async_wrapper

        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_error: Exception | None = None
                for attempt in range(max_attempts):
                    try:
                        if "api_key" in kwargs:
                            kwargs["api_key"] = api_config.get_current_api_key()
                        result = func(*args, **kwargs)
                        api_config.reset_to_primary()
                        return result
                    except Exception as error:
                        last_error = error
                        logger.error("Sync attempt %d/%d failed: %s", attempt + 1, max_attempts, error)
                        if not api_config.handle_api_error(error):
                            break
                        if attempt < max_attempts - 1:
                            time.sleep(min(2 ** attempt, 10))
                logger.error("All sync attempts failed. Last error: %s", last_error)
                raise last_error
            return sync_wrapper

    return decorator


# ── Convenience helpers (drop-in replacements for old module-level functions) ─

def get_current_api_key() -> str:
    return api_config.get_current_api_key()


def reset_to_primary_api() -> None:
    api_config.reset_to_primary()


def get_api_status() -> Dict[str, Any]:
    return api_config.get_status()


__all__ = [
    "api_config",
    "api_retry_with_fallback",
    "get_current_api_key",
    "reset_to_primary_api",
    "get_api_status",
]
