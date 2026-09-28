from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app import main as app_main
from app.digest import DigestPartialFailure
from tests.conftest import FakePublisher


async def test_non_rate_limited_failure_alerts_immediately(monkeypatch):
    monkeypatch.setattr(app_main, "run_digest", AsyncMock(side_effect=RuntimeError("boom")))
    sleep = AsyncMock()
    monkeypatch.setattr(app_main.asyncio, "sleep", sleep)
    publisher = FakePublisher()

    await app_main._run_digest_job(None, None, None, publisher, "chan")

    sleep.assert_not_awaited()
    assert len(publisher.plain) == 1
    assert "boom" in publisher.plain[0][1]


async def test_rate_limited_failure_retries_once_then_succeeds(monkeypatch):
    run_digest = AsyncMock(
        side_effect=[DigestPartialFailure(["x: quota"], rate_limited=True), None]
    )
    monkeypatch.setattr(app_main, "run_digest", run_digest)
    sleep = AsyncMock()
    monkeypatch.setattr(app_main.asyncio, "sleep", sleep)
    publisher = FakePublisher()

    await app_main._run_digest_job(None, None, None, publisher, "chan")

    sleep.assert_awaited_once_with(app_main.RATE_LIMIT_RETRY_SECONDS)
    assert run_digest.await_count == 2
    assert publisher.plain == []  # no alert: the retry succeeded


async def test_rate_limited_failure_alerts_after_retry_also_fails(monkeypatch):
    run_digest = AsyncMock(
        side_effect=[
            DigestPartialFailure(["x: quota"], rate_limited=True),
            DigestPartialFailure(["x: quota"], rate_limited=True),
        ]
    )
    monkeypatch.setattr(app_main, "run_digest", run_digest)
    sleep = AsyncMock()
    monkeypatch.setattr(app_main.asyncio, "sleep", sleep)
    publisher = FakePublisher()

    await app_main._run_digest_job(None, None, None, publisher, "chan")

    sleep.assert_awaited_once()  # retried exactly once, not in a loop
    assert run_digest.await_count == 2
    assert len(publisher.plain) == 1


async def test_alert_failure_is_swallowed(monkeypatch):
    """If even sending the alert fails, the job must not raise."""
    monkeypatch.setattr(app_main, "run_digest", AsyncMock(side_effect=RuntimeError("boom")))
    publisher = FakePublisher()
    publisher.send_plain = AsyncMock(side_effect=RuntimeError("telegram is also down"))

    await app_main._run_digest_job(None, None, None, publisher, "chan")  # must not raise
