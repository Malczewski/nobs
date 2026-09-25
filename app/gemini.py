"""Thin async wrapper around Gemini Flash with exponential backoff.

`google-generativeai` is synchronous, so calls are offloaded to a thread.
Rate-limit / transient errors (HTTP 429 and 5xx) are retried with exponential
backoff + jitter via tenacity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as gexc
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# Errors worth retrying: quota/rate limit, transient server, deadline.
_RETRYABLE = (
    gexc.ResourceExhausted,   # 429
    gexc.ServiceUnavailable,  # 503
    gexc.InternalServerError,  # 500
    gexc.DeadlineExceeded,    # 504
    gexc.TooManyRequests,
)

# Default backoff when the server gives no explicit hint. The free tier is
# ~5 requests/min, so retries need to wait on the order of tens of seconds.
_DEFAULT_RETRY_INITIAL = 15.0
_DEFAULT_RETRY_MAX = 300.0
_RETRY_ATTEMPTS = 8

# Added to the server-suggested retry_delay so we don't retry a hair too early.
_RETRY_BUFFER = 2.0

# Client-side throttle: minimum spacing between requests to stay under the
# free-tier 5 requests/min cap. 60/5 = 12s is the exact limit; 15s keeps us at
# <=4 requests in any rolling 60s window, leaving margin for clock skew and the
# occasional internal retry call (which does not pass through the throttle).
_MIN_REQUEST_INTERVAL = 15.0

# Hard ceiling on any single wait, so a bogus server hint can't stall us forever.
_MAX_RETRY_WAIT = 300.0


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """Extract the server-suggested retry delay (RetryInfo) from an error.

    Must never raise: it feeds the wait strategy, and a throw there would abort
    the retry loop. Any parsing problem falls through to None.
    """
    if exc is None:
        return None
    # Structured details (google.rpc.RetryInfo) when available.
    try:
        for detail in getattr(exc, "details", None) or []:
            retry_delay = getattr(detail, "retry_delay", None)
            if retry_delay is not None:
                seconds = retry_delay.seconds + retry_delay.nanos / 1e9
                if seconds > 0:
                    return float(seconds)
    except Exception:  # noqa: BLE001 - parsing is best-effort
        pass
    # Fallback: parse the textual representation (e.g. "retry_delay { seconds: 48 }").
    try:
        match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(exc))
        if match:
            return float(match.group(1))
    except Exception:  # noqa: BLE001 - parsing is best-effort
        pass
    return None


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient/per-minute errors, but not a per-day quota exhaustion.

    A daily quota (e.g. "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    won't reset within this process's backoff window, so retrying just burns
    attempts and time for a guaranteed failure. Fail fast instead so the
    caller (daily digest) can decide on a longer-scale retry.
    """
    if not isinstance(exc, _RETRYABLE):
        return False
    if "PerDay" in str(exc):
        return False
    return True


def _wait_for_retry(retry_state) -> float:
    """Wait the server-suggested delay if present, else exponential backoff.

    Always returns a sane, bounded float; never raises.
    """
    try:
        backoff = wait_exponential_jitter(
            initial=_DEFAULT_RETRY_INITIAL, max=_DEFAULT_RETRY_MAX
        )(retry_state)
    except Exception:  # noqa: BLE001 - fall back to a safe constant
        backoff = _DEFAULT_RETRY_INITIAL

    exc = None
    outcome = getattr(retry_state, "outcome", None)
    if outcome is not None and outcome.failed:
        exc = outcome.exception()

    server_delay = _retry_after_seconds(exc)
    wait = backoff if server_delay is None else max(backoff, server_delay + _RETRY_BUFFER)
    # Clamp to a hard ceiling regardless of server hint or backoff growth.
    return min(max(wait, 0.0), _MAX_RETRY_WAIT)


class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Serialize + space out requests to respect the per-minute quota.
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=_wait_for_retry,
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _generate_sync(self, model: str, prompt: str) -> str:
        gen_model = genai.GenerativeModel(model)
        response = gen_model.generate_content(prompt)
        return (response.text or "").strip()

    async def _throttle(self) -> None:
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = _MIN_REQUEST_INTERVAL - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    async def generate(self, model: str, prompt: str) -> str:
        await self._throttle()
        return await asyncio.to_thread(self._generate_sync, model, prompt)

    async def generate_json(self, model: str, prompt: str) -> Any:
        """Generate and parse a JSON response, tolerating markdown code fences."""
        text = await self.generate(model, prompt)
        return _parse_json(text)


def _parse_json(text: str) -> Any:
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: grab the first JSON array/object in the text.
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise
