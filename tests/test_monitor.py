from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app import monitor as mon
from app.monitor import ChannelMonitor, PendingMessage
from app.storage import SeenStore
from tests.conftest import FakeConfigLoader, FakeGemini, FakePublisher


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_resolve_source_numeric_id():
    assert mon._resolve_source("-1001234") == -1001234


def test_resolve_source_t_me_link():
    assert mon._resolve_source("https://t.me/somechannel") == "somechannel"


def test_resolve_source_at_username():
    assert mon._resolve_source("@somechannel") == "somechannel"


def test_resolve_source_bare_username():
    assert mon._resolve_source("somechannel") == "somechannel"


def test_strip_patterns_removes_footer_line():
    text = "Real content.\n\nУкраїна Online | Підписатись..."
    stripped = mon._strip_patterns(text, [r"^.*\|\s*Підписатись.*$"])
    assert "Підписатись" not in stripped
    assert "Real content." in stripped


def test_strip_patterns_ignores_invalid_regex():
    text = "Real content."
    # An unbalanced parenthesis is invalid regex; must not raise.
    stripped = mon._strip_patterns(text, ["("])
    assert stripped == "Real content."


def test_matches_skip_keyword_true():
    assert mon._matches_skip_keyword("Жовтий рівень небезпеки: тривога", ["Жовтий рівень небезпеки"])


def test_matches_skip_keyword_false():
    assert not mon._matches_skip_keyword("Some other news", ["Жовтий рівень небезпеки"])


def test_matches_skip_keyword_ignores_blank_entries():
    assert not mon._matches_skip_keyword("text", ["", None])  # type: ignore[list-item]


def test_build_batch_prompt_numbers_messages():
    pending = [
        PendingMessage(text="First", link="http://a"),
        PendingMessage(text="Second", link="http://b"),
    ]
    prompt = mon._build_batch_prompt("EVAL", "TRANSFORM", pending)
    assert "EVAL" in prompt
    assert "TRANSFORM" in prompt
    assert "[0] First" in prompt
    assert "[1] Second" in prompt


# ---------------------------------------------------------------------------
# ChannelMonitor — constructed without touching Telethon
# ---------------------------------------------------------------------------


@dataclass
class _FakeMessage:
    chat_id: int
    id: int
    message: str
    date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _FakeChat:
    username: str | None = None


def make_monitor(config_loader, gemini=None, store=None, publisher=None) -> ChannelMonitor:
    monitor = ChannelMonitor.__new__(ChannelMonitor)
    monitor._config_loader = config_loader
    monitor._gemini = gemini
    monitor._store = store
    monitor._publisher = publisher
    monitor._target_channel_id = "target-chan"
    monitor._entity = None
    monitor._backfill_task = None
    monitor._flush_task = None
    monitor._pending = []
    monitor._pending_lock = asyncio.Lock()
    return monitor


@pytest.fixture
def store(tmp_path: Path) -> SeenStore:
    return SeenStore(str(tmp_path / "seen.db"))


@pytest.fixture
def monitor(monitor_config, store, fake_gemini, fake_publisher) -> ChannelMonitor:
    config_loader = FakeConfigLoader(_wrap(monitor_config))
    return make_monitor(config_loader, gemini=fake_gemini, store=store, publisher=fake_publisher)


def _wrap(monitor_config):
    from app.config import AppConfig, DigestConfig

    return AppConfig(digest=DigestConfig(), monitor=monitor_config)


def test_prepare_skips_already_seen(monitor, store):
    store.mark_seen(mon.DEDUP_NAMESPACE, "1:1")
    msg = _FakeMessage(chat_id=1, id=1, message="News")
    assert monitor._prepare(msg, _FakeChat()) is None


def test_prepare_marks_seen_and_skips_empty_text(monitor, store):
    msg = _FakeMessage(chat_id=1, id=2, message="")
    assert monitor._prepare(msg, _FakeChat()) is None
    assert store.is_seen(mon.DEDUP_NAMESPACE, "1:2")


def test_prepare_marks_seen_and_skips_keyword_match(monitor, store):
    msg = _FakeMessage(chat_id=1, id=3, message="Жовтий рівень небезпеки: тривога у Києві")
    assert monitor._prepare(msg, _FakeChat()) is None
    assert store.is_seen(mon.DEDUP_NAMESPACE, "1:3")


def test_prepare_strips_footer_before_returning(monitor):
    msg = _FakeMessage(chat_id=1, id=4, message="Real news.\n\nУкраїна Online | Підписатись...")
    pending = monitor._prepare(msg, _FakeChat())
    assert pending is not None
    assert "Підписатись" not in pending.text
    assert "Real news." in pending.text


def test_prepare_returns_none_when_disabled(monitor_config, store, fake_gemini, fake_publisher):
    monitor_config.enabled = False
    config_loader = FakeConfigLoader(_wrap(monitor_config))
    monitor = make_monitor(config_loader, gemini=fake_gemini, store=store, publisher=fake_publisher)

    msg = _FakeMessage(chat_id=1, id=5, message="News")
    assert monitor._prepare(msg, _FakeChat()) is None


def test_prepare_builds_public_channel_link(monitor):
    msg = _FakeMessage(chat_id=1, id=6, message="News")
    pending = monitor._prepare(msg, _FakeChat(username="mychannel"))
    assert pending.link == "https://t.me/mychannel/6"


def test_prepare_builds_private_channel_link(monitor):
    msg = _FakeMessage(chat_id=-1001234567890, id=7, message="News")
    pending = monitor._prepare(msg, _FakeChat(username=None))
    assert pending.link == "https://t.me/c/1234567890/7"


async def test_on_new_message_enqueues_pending(monitor):
    class _FakeEvent:
        def __init__(self, message):
            self.message = message

        async def get_chat(self):
            return _FakeChat(username="mychannel")

    msg = _FakeMessage(chat_id=1, id=8, message="News")
    await monitor._on_new_message(_FakeEvent(msg))

    assert len(monitor._pending) == 1
    assert monitor._pending[0].text == "News"


async def test_on_new_message_drops_filtered_message(monitor):
    class _FakeEvent:
        def __init__(self, message):
            self.message = message

        async def get_chat(self):
            return _FakeChat()

    msg = _FakeMessage(chat_id=1, id=9, message="Жовтий рівень небезпеки")
    await monitor._on_new_message(_FakeEvent(msg))

    assert monitor._pending == []


# ---------------------------------------------------------------------------
# _process_batch / _forward
# ---------------------------------------------------------------------------


async def test_process_batch_forwards_kept_items(monitor, fake_gemini, fake_publisher):
    pending = [PendingMessage(text="Raw 1", link="http://a"), PendingMessage(text="Raw 2", link="http://b")]
    fake_gemini.queue([{"text": "Combined summary", "sources": [0, 1]}])

    await monitor._process_batch("model", "eval", "transform", pending)

    assert len(fake_publisher.sent) == 1
    body = fake_publisher.sent[0][1]
    assert "Combined summary" in body
    assert "http://a" in body
    assert "http://b" in body


async def test_process_batch_drops_items_with_empty_text(monitor, fake_gemini, fake_publisher):
    pending = [PendingMessage(text="Raw", link="http://a")]
    fake_gemini.queue([{"text": "  ", "sources": [0]}])

    await monitor._process_batch("model", "eval", "transform", pending)

    assert fake_publisher.sent == []


async def test_process_batch_handles_empty_result(monitor, fake_gemini, fake_publisher):
    pending = [PendingMessage(text="Raw", link="http://a")]
    fake_gemini.queue([])

    await monitor._process_batch("model", "eval", "transform", pending)

    assert fake_publisher.sent == []


async def test_process_batch_swallows_gemini_failure(monitor, fake_gemini, fake_publisher):
    pending = [PendingMessage(text="Raw", link="http://a")]
    fake_gemini.queue(RuntimeError("gemini is down"))

    await monitor._process_batch("model", "eval", "transform", pending)  # must not raise

    assert fake_publisher.sent == []


async def test_process_batch_handles_non_list_response(monitor, fake_gemini, fake_publisher):
    pending = [PendingMessage(text="Raw", link="http://a")]
    fake_gemini.queue({"unexpected": "shape"})

    await monitor._process_batch("model", "eval", "transform", pending)  # must not raise

    assert fake_publisher.sent == []


async def test_process_batch_ignores_out_of_range_source_indices(monitor, fake_gemini, fake_publisher):
    pending = [PendingMessage(text="Raw", link="http://a")]
    fake_gemini.queue([{"text": "Combined", "sources": [0, 99]}])

    await monitor._process_batch("model", "eval", "transform", pending)

    body = fake_publisher.sent[0][1]
    assert body.count("🔗") == 1


async def test_forward_single_link(monitor, fake_publisher):
    await monitor._forward("Body text", links=["http://a"])
    body = fake_publisher.sent[0][1]
    assert '<a href="http://a">Original</a>' in body


async def test_forward_multiple_links_numbered(monitor, fake_publisher):
    await monitor._forward("Body text", links=["http://a", "http://b"])
    body = fake_publisher.sent[0][1]
    assert "Джерело 1" in body
    assert "Джерело 2" in body


async def test_forward_no_links(monitor, fake_publisher):
    await monitor._forward("Body text", links=None)
    body = fake_publisher.sent[0][1]
    assert "🔗" not in body


async def test_forward_escapes_html_in_text(monitor, fake_publisher):
    await monitor._forward("<script>alert(1)</script>", links=None)
    body = fake_publisher.sent[0][1]
    assert "<script>" not in body


# ---------------------------------------------------------------------------
# _flush
# ---------------------------------------------------------------------------


async def test_flush_is_noop_when_pending_empty(monitor, fake_gemini):
    await monitor._flush()
    assert fake_gemini.calls == []


async def test_flush_processes_and_clears_pending(monitor, fake_gemini, fake_publisher):
    monitor._pending = [PendingMessage(text="Raw", link="http://a")]
    fake_gemini.queue([{"text": "Combined", "sources": [0]}])

    await monitor._flush()

    assert monitor._pending == []
    assert len(fake_publisher.sent) == 1


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------


async def test_disconnect_flushes_pending_and_disconnects_client(monitor, fake_gemini, fake_publisher):
    monitor._pending = [PendingMessage(text="Raw", link="http://a")]
    fake_gemini.queue([{"text": "Combined", "sources": [0]}])
    monitor._client = AsyncMock()

    await monitor.disconnect()

    assert len(fake_publisher.sent) == 1  # pending batch wasn't lost on shutdown
    monitor._client.disconnect.assert_awaited_once()
