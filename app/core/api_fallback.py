import asyncio
import inspect
import logging
import threading
import time
from typing import Any, Callable, Dict
from functools import wraps

from app.core.config import settings

logger = logging.getLogger(__name__)

# Ordered fallback chain — index 0 is tried first.
_TIER_NAMES = ("primary", "secondary", "tertiary", "fourth", "fifth")

class APIConfig:
    """Text API configuration with N-tier fallback support. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current_index: int = 0  

        self._tiers: Dict[str, Dict[str, Any]] = {
            "primary": {
                "api_key":     settings.PRIMARY_GEMINI_API_KEY.get_secret_value(),
                "name":        "Primary Gemini API",
                "retry_count": 0,
                "max_retries": 3,
            },
            "secondary": {
                "api_key":     settings.SECONDARY_GEMINI_API_KEY.get_secret_value(),
                "name":        "Secondary Gemini API",
                "retry_count": 0,
                "max_retries": 3,
            },
            "tertiary": {
                "api_key":     settings.TERTIARY_GEMINI_API_KEY.get_secret_value(),
                "name":        "Tertiary Gemini API",
                "retry_count": 0,
                "max_retries": 3,
            },
            "fourth": {
                "api_key":     settings.FOURTH_GEMINI_API_KEY.get_secret_value(),
                "name":        "Fourth Gemini API",
                "retry_count": 0,
                "max_retries": 3,
            },
            "fifth": {
                "api_key":     settings.FIFTH_GEMINI_API_KEY.get_secret_value(),
                "name":        "Fifth Gemini API",
                "retry_count": 0,
                "max_retries": 3,
            },
        }

    @property
    def _current_tier(self) -> str:
        return _TIER_NAMES[self._current_index]

    @property
    def _current_cfg(self) -> Dict[str, Any]:
        return self._tiers[self._current_tier]

    def get_current_api_key(self) -> str:
        with self._lock:
            return self._current_cfg["api_key"]

    def get_status(self) -> Dict[str, Any]:
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
        with self._lock:
            if self._current_index != 0:
                self._current_index = 0
                for cfg in self._tiers.values():
                    cfg["retry_count"] = 0
                logger.info("Reset to primary Gemini API")

    def handle_api_error(self, error: Exception) -> bool:
        with self._lock:
            cfg = self._current_cfg
            cfg["retry_count"] += 1
            
            if cfg["retry_count"] < cfg["max_retries"]:
                return True 

            next_index = self._current_index + 1
            if next_index < len(_TIER_NAMES):
                self._current_index = next_index
                self._current_cfg["retry_count"] = 0 
                logger.warning("Switched to %s", self._current_cfg["name"])
                return True

            return False

api_config = APIConfig()

def api_retry_with_fallback(max_attempts: int = 15):
    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_error: Exception | None = None
                for attempt in range(max_attempts):
                    try:
                        if "api_key" in kwargs:
                            kwargs["api_key"] = api_config.get_current_api_key()
                        
                        return await func(*args, **kwargs)
                    except Exception as error:
                        last_error = error
                        if not api_config.handle_api_error(error):
                            api_config.reset_to_primary() 
                            break
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(min(2 ** attempt, 10))
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
                        return func(*args, **kwargs)
                    except Exception as error:
                        last_error = error
                        if not api_config.handle_api_error(error):
                            api_config.reset_to_primary()
                            break
                        if attempt < max_attempts - 1:
                            time.sleep(min(2 ** attempt, 10))
                raise last_error
            return sync_wrapper
    return decorator

# Convenience helpers
def get_current_api_key() -> str:
    return api_config.get_current_api_key()

def reset_to_primary_api() -> None:
    api_config.reset_to_primary()

def get_api_status() -> Dict[str, Any]:
    return api_config.get_status()

__all__ = ["api_config", "api_retry_with_fallback", "get_current_api_key", "reset_to_primary_api", "get_api_status"]