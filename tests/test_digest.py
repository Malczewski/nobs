from __future__ import annotations

import time
from pathlib import Path

import pytest
from google.api_core import exceptions as gexc

from app import digest as dg
from app.config import DigestConfig, Source
from tests.conftest import FakeConfigLoader, FakeGemini, FakePublisher


def _raw_entry(title="T", link="http://x/1", summary="S", published_epoch=None, extra_id=None):
    entry = {"title": title, "link": link, "summary": summary}
    if published_epoch is not None:
        entry["published_parsed"] = time.gmtime(published_epoch)
    if extra_id is not None:
        entry["id"] = extra_id
    return entry


class _FakeParsed:
    def __init__(self, entries, bozo=False, bozo_exception=None):
        self.entries = entries
        self.bozo = bozo
        self.bozo_exception = bozo_exception


# ---------------------------------------------------------------------------
# fetch_feed
# ---------------------------------------------------------------------------


def test_fetch_feed_keeps_recent_entries(monkeypatch):
    now = time.time()
    parsed = _FakeParsed([_raw_entry(published_epoch=now - 3600)])
    monkeypatch.setattr(dg.feedparser, "parse", lambda url: parsed)

    source = Source(url="http://x", label="L", prompt="p", lookback_hours=24)
    entries = dg.fetch_feed(source)

    assert len(entries) == 1
    assert entries[0].title == "T"


def test_fetch_feed_drops_entries_older_than_lookback(monkeypatch):
    now = time.time()
    parsed = _FakeParsed([_raw_entry(published_epoch=now - 48 * 3600)])
    monkeypatch.setattr(dg.feedparser, "parse", lambda url: parsed)

    source = Source(url="http://x", label="L", prompt="p", lookback_hours=24)
    entries = dg.fetch_feed(source)

    assert entries == []


def test_fetch_feed_keeps_entries_without_a_date(monkeypatch):
    parsed = _FakeParsed([_raw_entry(published_epoch=None)])
    monkeypatch.setattr(dg.feedparser, "parse", lambda url: parsed)

    source = Source(url="http://x", label="L", prompt="p", lookback_hours=24)
    entries = dg.fetch_feed(source)

    assert len(entries) == 1


def test_fetch_feed_respects_max_items(monkeypatch):
    now = time.time()
    parsed = _FakeParsed(
        [_raw_entry(title=f"T{i}", published_epoch=now, extra_id=str(i)) for i in range(5)]
    )
    monkeypatch.setattr(dg.feedparser, "parse", lambda url: parsed)

    source = Source(url="http://x", label="L", prompt="p", max_items=2)
    entries = dg.fetch_feed(source)

    assert len(entries) == 2


def test_fetch_feed_entry_id_falls_back_to_title(monkeypatch):
    parsed = _FakeParsed([{"title": "Only a title"}])
    monkeypatch.setattr(dg.feedparser, "parse", lambda url: parsed)

    source = Source(url="http://x", label="L", prompt="p")
    entries = dg.fetch_feed(source)

    assert entries[0].entry_id == "Only a title"


# ---------------------------------------------------------------------------
# _build_prompt / _render_message
# ---------------------------------------------------------------------------


def test_build_prompt_includes_all_pieces():
    entry = dg.FeedEntry(entry_id="1", title="Title", link="http://x", summary="Summary")
    prompt = dg._build_prompt(
        Source(url="u", label="L", prompt="SOURCE_PROMPT"),
        "Ukrainian",
        "SUMMARIZE_PROMPT",
        [entry],
    )
    assert "SOURCE_PROMPT" in prompt
    assert "SUMMARIZE_PROMPT" in prompt
    assert "Ukrainian" in prompt
    assert "Title" in prompt
    assert "http://x" in prompt


def test_render_message_escapes_html():
    items = [{"title": "<b>bold</b>", "link": "http://x", "summary": "S & T"}]
    rendered = dg._render_message("Label", items)
    assert "<b>bold</b>" not in rendered  # the malicious title must be escaped
    assert "&lt;b&gt;bold&lt;/b&gt;" in rendered
    assert "S &amp; T" in rendered
    assert '<a href="http://x">' in rendered


def test_render_message_without_link_or_summary():
    items = [{"title": "Just a title"}]
    rendered = dg._render_message("Label", items)
    assert "• Just a title" in rendered
    assert "<a href" not in rendered


# ---------------------------------------------------------------------------
# process_source
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path):
    from app.storage import SeenStore

    return SeenStore(str(tmp_path / "seen.db"))


async def test_process_source_skips_when_no_fresh_entries(monkeypatch, store, fake_gemini):
    monkeypatch.setattr(dg, "fetch_feed", lambda source: [dg.FeedEntry("1", "T", "L", "S")])
    store.mark_seen(dg.DEDUP_NAMESPACE, "1")

    result = await dg.process_source(
        Source(url="u", label="L", prompt="p"),
        language="Ukrainian",
        model="m",
        summarize_prompt="sp",
        gemini=fake_gemini,
        store=store,
    )

    assert result is None
    assert fake_gemini.calls == []  # never worth calling the LLM


async def test_process_source_renders_message_and_marks_seen(monkeypatch, store, fake_gemini):
    monkeypatch.setattr(dg, "fetch_feed", lambda source: [dg.FeedEntry("1", "T", "L", "S")])
    fake_gemini.queue([{"title": "T", "link": "L", "summary": "S"}])

    result = await dg.process_source(
        Source(url="u", label="Label", prompt="p"),
        language="Ukrainian",
        model="m",
        summarize_prompt="sp",
        gemini=fake_gemini,
        store=store,
    )

    assert result is not None
    assert "Label" in result
    assert store.is_seen(dg.DEDUP_NAMESPACE, "1")


async def test_process_source_returns_none_on_empty_model_output(monkeypatch, store, fake_gemini):
    monkeypatch.setattr(dg, "fetch_feed", lambda source: [dg.FeedEntry("1", "T", "L", "S")])
    fake_gemini.queue([])

    result = await dg.process_source(
        Source(url="u", label="L", prompt="p"),
        language="Ukrainian",
        model="m",
        summarize_prompt="sp",
        gemini=fake_gemini,
        store=store,
    )

    assert result is None
    # Still marked seen, so we don't reconsider it tomorrow.
    assert store.is_seen(dg.DEDUP_NAMESPACE, "1")


async def test_process_source_does_not_mark_seen_on_gemini_failure(monkeypatch, store, fake_gemini):
    monkeypatch.setattr(dg, "fetch_feed", lambda source: [dg.FeedEntry("1", "T", "L", "S")])
    fake_gemini.queue(gexc.ResourceExhausted("boom"))

    with pytest.raises(gexc.ResourceExhausted):
        await dg.process_source(
            Source(url="u", label="L", prompt="p"),
            language="Ukrainian",
            model="m",
            summarize_prompt="sp",
            gemini=fake_gemini,
            store=store,
        )

    # Not marked seen: worth reconsidering the entry on the next run.
    assert not store.is_seen(dg.DEDUP_NAMESPACE, "1")


# ---------------------------------------------------------------------------
# run_digest
# ---------------------------------------------------------------------------


async def test_run_digest_posts_messages_for_each_source(monkeypatch, digest_config, fake_publisher, store):
    digest_config.sources = [
        Source(url="u1", label="A", prompt="p"),
        Source(url="u2", label="B", prompt="p"),
    ]
    config_loader = FakeConfigLoader(_wrap(digest_config))

    async def fake_process_source(source, **kwargs):
        return f"message for {source.label}"

    monkeypatch.setattr(dg, "process_source", fake_process_source)

    await dg.run_digest(
        config_loader=config_loader,
        gemini=None,
        store=store,
        publisher=fake_publisher,
        channel_id="chan",
    )

    assert len(fake_publisher.sent) == 2
    assert fake_publisher.sent[0][1] == "message for A"


async def test_run_digest_raises_partial_failure_and_keeps_going(
    monkeypatch, digest_config, fake_publisher, store
):
    digest_config.sources = [
        Source(url="u1", label="A", prompt="p"),
        Source(url="u2", label="B", prompt="p"),
    ]
    config_loader = FakeConfigLoader(_wrap(digest_config))

    async def fake_process_source(source, **kwargs):
        if source.label == "A":
            raise RuntimeError("boom")
        return "message for B"

    monkeypatch.setattr(dg, "process_source", fake_process_source)

    with pytest.raises(dg.DigestPartialFailure) as exc_info:
        await dg.run_digest(
            config_loader=config_loader,
            gemini=None,
            store=store,
            publisher=fake_publisher,
            channel_id="chan",
        )

    assert "A" in str(exc_info.value)
    assert len(fake_publisher.sent) == 1  # B still got posted
    assert exc_info.value.rate_limited is False


async def test_run_digest_flags_rate_limited_failures(monkeypatch, digest_config, fake_publisher, store):
    digest_config.sources = [Source(url="u1", label="A", prompt="p")]
    config_loader = FakeConfigLoader(_wrap(digest_config))

    async def fake_process_source(source, **kwargs):
        raise gexc.ResourceExhausted("quota exceeded")

    monkeypatch.setattr(dg, "process_source", fake_process_source)

    with pytest.raises(dg.DigestPartialFailure) as exc_info:
        await dg.run_digest(
            config_loader=config_loader,
            gemini=None,
            store=store,
            publisher=fake_publisher,
            channel_id="chan",
        )

    assert exc_info.value.rate_limited is True


def _wrap(digest_config: DigestConfig):
    from app.config import AppConfig, MonitorConfig

    return AppConfig(digest=digest_config, monitor=MonitorConfig())
