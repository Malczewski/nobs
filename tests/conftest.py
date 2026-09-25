"""Shared test fixtures.

Nothing here talks to a real network: GCS, Gemini and Telegram are all
replaced by small fakes that implement just the interface the app code
actually calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import AppConfig, DigestConfig, MonitorConfig, Source


class FakeConfigLoader:
    """Drop-in for ConfigLoader that returns a fixed, in-memory AppConfig."""

    def __init__(self, config: AppConfig):
        self._config = config

    def get(self, *, force: bool = False) -> AppConfig:
        return self._config


class FakeGemini:
    """Queue of generate_json results/exceptions, popped in call order."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self._results: list[Any] = []

    def queue(self, result: Any) -> None:
        """Queue a return value, or an exception instance to be raised."""
        self._results.append(result)

    async def generate_json(self, model: str, prompt: str) -> Any:
        self.calls.append((model, prompt))
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakePublisher:
    """Records every message instead of hitting Telegram."""

    def __init__(self):
        self.sent: list[tuple[str, str, bool]] = []
        self.plain: list[tuple[str, str]] = []

    async def send(self, chat_id: str, text: str, *, disable_preview: bool = False) -> None:
        self.sent.append((chat_id, text, disable_preview))

    async def send_plain(self, chat_id: str, text: str) -> None:
        self.plain.append((chat_id, text))


@pytest.fixture
def digest_config() -> DigestConfig:
    return DigestConfig(
        hour=8,
        minute=0,
        timezone="UTC",
        language="Ukrainian",
        model="gemini-test",
        summarize_prompt="Summarize.",
        sources=[
            Source(url="http://example.com/feed", label="Test", prompt="Extract news."),
        ],
    )


@pytest.fixture
def monitor_config() -> MonitorConfig:
    return MonitorConfig(
        enabled=True,
        model="gemini-test",
        evaluate_prompt="Filter.",
        transform_prompt="Rewrite.",
        skip_keywords=["Жовтий рівень небезпеки"],
        strip_patterns=[r"^.*\|\s*Підписатись.*$"],
        batch_window_minutes=5.0,
    )


@pytest.fixture
def fake_gemini() -> FakeGemini:
    return FakeGemini()


@pytest.fixture
def fake_publisher() -> FakePublisher:
    return FakePublisher()
