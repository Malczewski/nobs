from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from telegram.error import RetryAfter, TimedOut

from app.telegram_bot import Publisher, _split_message


# ---------------------------------------------------------------------------
# _split_message
# ---------------------------------------------------------------------------


def test_split_message_short_text_is_one_chunk():
    assert _split_message("hello") == ["hello"]


def test_split_message_splits_on_line_boundaries():
    lines = [f"{i}-" + "x" * 3000 for i in range(3)]  # distinct, ~9000 chars total
    chunks = _split_message("\n".join(lines))
    assert len(chunks) > 1
    assert all(len(chunk) <= 4096 for chunk in chunks)
    # Every line survives whole in exactly one chunk (none dropped or corrupted).
    for original_line in lines:
        assert sum(chunk.count(original_line) for chunk in chunks) == 1


def test_split_message_hard_splits_a_single_line_longer_than_limit():
    line = "x" * 10000
    chunks = _split_message(line, limit=4096)
    assert all(len(chunk) <= 4096 for chunk in chunks)
    assert "".join(chunks) == line


def test_split_message_respects_custom_limit():
    text = "aaaa\nbbbb\ncccc"
    chunks = _split_message(text, limit=10)
    assert all(len(c) <= 10 for c in chunks)


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


@pytest.fixture
def publisher() -> Publisher:
    pub = Publisher("fake-token")
    pub._bot = AsyncMock()
    return pub


async def test_send_one_happy_path(publisher):
    await publisher.send("chat", "hello", disable_preview=True)
    publisher._bot.send_message.assert_awaited_once()
    _, kwargs = publisher._bot.send_message.call_args
    assert kwargs["chat_id"] == "chat"
    assert kwargs["text"] == "hello"
    assert kwargs["disable_web_page_preview"] is True


async def test_send_splits_long_messages_into_multiple_calls(publisher):
    text = "\n".join(["x" * 3000] * 3)
    await publisher.send("chat", text)
    assert publisher._bot.send_message.await_count > 1


async def test_send_one_retries_after_rate_limit(publisher, monkeypatch):
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr("app.telegram_bot.asyncio.sleep", fake_sleep)
    publisher._bot.send_message = AsyncMock(side_effect=[RetryAfter(3), None])

    await publisher.send("chat", "hello")

    assert sleep_calls == [4]  # retry_after + 1
    assert publisher._bot.send_message.await_count == 2


async def test_send_one_retries_once_on_timeout(publisher):
    publisher._bot.send_message = AsyncMock(side_effect=[TimedOut(), None])

    await publisher.send("chat", "hello")

    assert publisher._bot.send_message.await_count == 2


async def test_send_plain_does_not_pass_html_parse_mode(publisher):
    await publisher.send_plain("chat", "hello")
    _, kwargs = publisher._bot.send_message.call_args
    assert "parse_mode" not in kwargs
