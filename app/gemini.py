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
import threading
import time
from collections import deque
from datetime import datetime, time as dtime, timedelta
from typing import Any

import google.generativeai as genai
import pytz
from google.api_core import exceptions as gexc
from tenacity import (
    Retrying,
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


# Substrings identifying a 429 that won't clear on its own: a fixed quota/
# balance that only changes via manual action (wait a day, or fix billing),
# never via a short backoff. Matched against str(exc) since these come back
# as plain error text, not a structured, reliably-typed field.
_UNRECOVERABLE_MARKERS = (
    "PerDay",  # e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier
    "prepayment credits are depleted",  # billing balance exhausted, needs a top-up
)


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient/per-minute errors, but not a quota/billing dead end.

    A per-day quota or a depleted prepayment balance won't clear within this
    process's backoff window, so retrying just burns attempts and time for a
    guaranteed failure. Fail fast instead so the caller (daily digest) can
    decide on a longer-scale retry.
    """
    if not isinstance(exc, _RETRYABLE):
        return False
    text = str(exc)
    if any(marker in text for marker in _UNRECOVERABLE_MARKERS):
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


# Gemini free-tier daily quotas reset at midnight Pacific time.
_QUOTA_TZ = pytz.timezone("America/Los_Angeles")


def _quota_details(exc: BaseException) -> dict[str, Any]:
    """Pull quota_id / quota_value out of the textual 429 error (best-effort)."""
    text = str(exc)
    quota_id = re.search(r'quota_id:\s*"([^"]+)"', text)
    quota_value = re.search(r"quota_value:\s*(\d+)", text)
    return {
        "quota_id": quota_id.group(1) if quota_id else None,
        "limit": int(quota_value.group(1)) if quota_value else None,
        "retry_after": _retry_after_seconds(exc),
    }


def _quota_kind(exc: BaseException) -> str:
    text = str(exc)
    if "prepayment credits are depleted" in text:
        return "billing"
    quota_id = _quota_details(exc)["quota_id"] or text
    if "PerDay" in quota_id:
        return "per_day"
    if "PerMinute" in quota_id:
        return "per_minute"
    return "other"


def _until_quota_reset() -> str:
    now = datetime.now(_QUOTA_TZ)
    midnight = datetime.combine(now.date() + timedelta(days=1), dtime.min)
    minutes = int((_QUOTA_TZ.localize(midnight) - now).total_seconds() // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"


class GeminiQuotaError(gexc.ResourceExhausted):
    """A 429 with a human-readable message instead of the raw Google dump.

    Subclasses ResourceExhausted so existing `except`/isinstance checks keep
    working. The original error is chained as __cause__.
    """

    def __init__(self, message: str, *, kind: str):
        super().__init__(message)
        self.message = message
        self.kind = kind

    def __str__(self) -> str:
        return self.message


def _quota_message(
    exc: BaseException, model: str | None, local_today: int | None
) -> tuple[str, str]:
    """Return (kind, friendly message) for a 429 error."""
    kind = _quota_kind(exc)
    details = _quota_details(exc)
    on_model = f" for {model}" if model else ""
    limit = f" ({details['limit']}/{'day' if kind == 'per_day' else 'min'})" if details["limit"] else ""
    if kind == "per_day":
        msg = (
            f"Gemini daily request quota is used up{on_model}{limit}. "
            f"It resets at midnight Pacific time, in about {_until_quota_reset()}."
        )
    elif kind == "per_minute":
        msg = f"Gemini per-minute rate limit was exceeded{on_model}{limit}, even after retries."
    elif kind == "billing":
        msg = "Gemini prepaid balance is depleted. Top it up in Google AI Studio."
    else:
        msg = f"Gemini quota exceeded{on_model}{limit}."
    if local_today is not None and kind in ("per_day", "per_minute"):
        msg += (
            f" This bot has sent {local_today} request(s) today (its own count); "
            "if that is far below the limit, the API key/project is used by something else too."
        )
    return kind, msg


def describe_error(exc: BaseException) -> str:
    """Short, user-facing description of a failure; never a raw Google dump."""
    if isinstance(exc, GeminiQuotaError):
        return str(exc)
    if isinstance(exc, (gexc.ResourceExhausted, gexc.TooManyRequests)):
        return _quota_message(exc, None, None)[1]
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:200]}" + ("…" if len(text) > 200 else "")


class _UsageTracker:
    """Local count of requests per model per Pacific-time day.

    The Gemini API does not report remaining quota, so this is our own count of
    requests *this process* made (each retry counts). Comparing it with the
    limit from a 429 shows whether something else uses the same project.
    Resets on restart.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: str = ""
        self._today: dict[str, int] = {}
        self._recent: dict[str, deque[float]] = {}
        self.daily_limits: dict[str, int] = {}

    def record(self, model: str) -> tuple[int, int]:
        """Count one request. Returns (requests today, requests in last 60s)."""
        now = time.monotonic()
        day = datetime.now(_QUOTA_TZ).strftime("%Y-%m-%d")
        with self._lock:
            if day != self._day:
                self._day, self._today = day, {}
            self._today[model] = self._today.get(model, 0) + 1
            recent = self._recent.setdefault(model, deque())
            recent.append(now)
            while recent and now - recent[0] > 60:
                recent.popleft()
            return self._today[model], len(recent)

    def today(self, model: str) -> int:
        with self._lock:
            return self._today.get(model, 0)

    @property
    def day(self) -> str:
        return self._day


def _usage_fields(response: Any) -> str:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return "tokens=?"
    return (
        f"tokens_in={getattr(usage, 'prompt_token_count', '?')} "
        f"tokens_out={getattr(usage, 'candidates_token_count', '?')} "
        f"tokens_total={getattr(usage, 'total_token_count', '?')}"
    )


def _finish_reason(response: Any) -> str:
    try:
        reason = response.candidates[0].finish_reason
        return getattr(reason, "name", str(reason))
    except Exception:  # noqa: BLE001 - purely informational
        return "?"


class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Serialize + space out requests to respect the per-minute quota.
        self._rate_lock = asyncio.Lock()
        self._last_request_at = 0.0
        self._usage = _UsageTracker()

    def _call_once(self, model: str, prompt: str, label: str, attempt: int) -> str:
        """One real API request, with full logging of its outcome."""
        started = time.monotonic()
        base = f"label={label or '-'} model={model} attempt={attempt} prompt_chars={len(prompt)}"
        try:
            response = genai.GenerativeModel(model).generate_content(prompt)
        except Exception as exc:
            today, last_minute = self._usage.record(model)
            elapsed = time.monotonic() - started
            local = f"local_today={today} local_last_60s={last_minute}"
            if isinstance(exc, (gexc.ResourceExhausted, gexc.TooManyRequests)):
                details = _quota_details(exc)
                if details["limit"] and _quota_kind(exc) == "per_day":
                    self._usage.daily_limits[model] = details["limit"]
                logger.warning(
                    "Gemini request FAILED (429 %s) %s latency=%.1fs quota_id=%s limit=%s "
                    "retry_after=%s %s (local count: this process only, resets on restart)",
                    _quota_kind(exc), base, elapsed, details["quota_id"], details["limit"],
                    details["retry_after"], local,
                )
            else:
                logger.warning(
                    "Gemini request FAILED %s latency=%.1fs error=%s %s",
                    base, elapsed, describe_error(exc), local,
                )
            raise

        today, last_minute = self._usage.record(model)
        limit = self._usage.daily_limits.get(model)
        text = (response.text or "").strip()
        logger.info(
            "Gemini request ok %s latency=%.1fs %s finish=%s response_chars=%d "
            "local_today=%d%s local_last_60s=%d (quota day PT %s)",
            base, time.monotonic() - started, _usage_fields(response),
            _finish_reason(response), len(text), today,
            f"/{limit}" if limit else "", last_minute, self._usage.day,
        )
        return text

    def _generate_sync(self, model: str, prompt: str, label: str = "") -> str:
        attempt = 0

        def _attempt() -> str:
            nonlocal attempt
            attempt += 1
            return self._call_once(model, prompt, label, attempt)

        def _log_retry(retry_state) -> None:
            exc = retry_state.outcome.exception()
            logger.warning(
                "Gemini retry label=%s: attempt %d/%d failed (%s); waiting %.0fs",
                label or "-", retry_state.attempt_number, _RETRY_ATTEMPTS,
                describe_error(exc), retry_state.next_action.sleep,
            )

        retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            wait=_wait_for_retry,
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            before_sleep=_log_retry,
            reraise=True,
        )
        try:
            return retrying(_attempt)
        except (gexc.ResourceExhausted, gexc.TooManyRequests) as exc:
            if isinstance(exc, GeminiQuotaError):
                raise
            kind, message = _quota_message(exc, model, self._usage.today(model))
            raise GeminiQuotaError(message, kind=kind) from exc

    async def _throttle(self) -> None:
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = _MIN_REQUEST_INTERVAL - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    async def generate(self, model: str, prompt: str, label: str = "") -> str:
        await self._throttle()
        return await asyncio.to_thread(self._generate_sync, model, prompt, label)

    async def generate_json(self, model: str, prompt: str, label: str = "") -> Any:
        """Generate and parse a JSON response, tolerating markdown code fences."""
        text = await self.generate(model, prompt, label)
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
