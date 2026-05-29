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
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as gexc
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
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


class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential_jitter(initial=2, max=120),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _generate_sync(self, model: str, prompt: str) -> str:
        gen_model = genai.GenerativeModel(model)
        response = gen_model.generate_content(prompt)
        return (response.text or "").strip()

    async def generate(self, model: str, prompt: str) -> str:
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
