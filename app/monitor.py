"""Purpose 2 — channel monitor.

A Telethon user client listens for new messages in TELEGRAM_SOURCE_CHANNEL.
Messages are cheaply pre-filtered (keyword skip-list, footer stripping), then
buffered for `monitor.batch_window_minutes` and evaluated together in a single
Gemini call: near-duplicate updates (e.g. several posts about the same
bombardment) are combined into one entry, and everything that doesn't pass the
filter is dropped. Batching keeps Gemini usage roughly constant regardless of
how chatty the source channel gets, instead of one call per message.

Config (prompts, keyword/pattern lists) is re-read per batch — but TTL-cached
in ConfigLoader so edits in GCS apply within ~1 minute without a restart.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from telethon import TelegramClient, events
from telethon.sessions import StringSession

from .config import ConfigLoader
from .gemini import GeminiClient
from .storage import SeenStore
from .telegram_bot import Publisher

logger = logging.getLogger(__name__)

DEDUP_NAMESPACE = "tg_monitor"

# On startup, look back this many hours and process any messages we missed
# while down. Dedup (SeenStore) makes this idempotent across restarts.
BACKFILL_HOURS = 12


@dataclass
class PendingMessage:
    text: str
    link: str | None


def _resolve_source(value: str):
    """Accept a numeric channel id, a t.me username, or a bare @username."""
    value = value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    if value.startswith("https://t.me/"):
        return value.rsplit("/", 1)[-1]
    return value.lstrip("@") if value.startswith("@") else value


def _strip_patterns(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        try:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
        except re.error:
            logger.warning("Invalid strip_patterns regex, skipping: %s", pattern)
    return text.strip()


def _matches_skip_keyword(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords if keyword)


def _build_batch_prompt(evaluate_prompt: str, transform_prompt: str, pending: list[PendingMessage]) -> str:
    items_block = "\n\n".join(f"[{i}] {m.text}" for i, m in enumerate(pending))
    return (
        f"{evaluate_prompt}\n\n"
        f"{transform_prompt}\n\n"
        "You will receive a numbered batch of raw channel messages collected over "
        "the last few minutes. For each message, apply the filter above. Combine "
        "messages that report on the same event or topic (e.g. several updates "
        "about the same attack) into a single entry — render it as a short "
        "bullet list if it covers multiple distinct facts. Drop everything that "
        "doesn't pass the filter.\n\n"
        "Respond with ONLY a JSON array of objects, each with exactly:\n"
        '  "text": the rewritten/combined message text (no preamble, no labels),\n'
        '  "sources": a list of the integer indices it was built from.\n'
        "If nothing qualifies, return [].\n\n"
        f"Messages:\n{items_block}"
    )


class ChannelMonitor:
    def __init__(
        self,
        *,
        api_id: int,
        api_hash: str,
        session_string: str,
        source_channel: str,
        config_loader: ConfigLoader,
        gemini: GeminiClient,
        store: SeenStore,
        publisher: Publisher,
        target_channel_id: str,
    ):
        self._client = TelegramClient(StringSession(session_string), api_id, api_hash)
        self._source_channel = _resolve_source(source_channel)
        self._config_loader = config_loader
        self._gemini = gemini
        self._store = store
        self._publisher = publisher
        self._target_channel_id = target_channel_id
        self._entity = None
        self._backfill_task: asyncio.Task | None = None
        self._flush_task: asyncio.Task | None = None
        self._pending: list[PendingMessage] = []
        self._pending_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._client.start()  # uses the StringSession; non-interactive
        # Prime the session's dialog/state cache. With a StringSession that has
        # never fetched dialogs, Telethon may not apply the update stream for a
        # channel (so NewMessage never fires) even though get_entity() resolves
        # it and the account is subscribed. get_dialogs() registers the channel
        # so its pushed updates are delivered to the handler.
        await self._client.get_dialogs()
        self._entity = await self._client.get_entity(self._source_channel)
        self._client.add_event_handler(
            self._on_new_message, events.NewMessage(chats=self._entity)
        )
        logger.info("Monitor listening on source channel: %s", self._source_channel)

        self._flush_task = asyncio.create_task(self._flush_loop())

        # Catch up on anything posted while we were down. Run it in the
        # background so realtime handling starts immediately; dedup prevents
        # double-processing if a message also arrives via the live stream.
        self._backfill_task = asyncio.create_task(self._backfill(BACKFILL_HOURS))

    async def run_forever(self) -> None:
        await self.start()
        await self._client.run_until_disconnected()

    def run_until_disconnected(self):
        """Awaitable that resolves when the underlying client disconnects."""
        return self._client.run_until_disconnected()

    async def _on_new_message(self, event: events.NewMessage.Event) -> None:
        msg_id = f"{event.message.chat_id}:{event.message.id}"
        logger.info("Received message %s", msg_id)
        chat = await event.get_chat()
        pending = self._prepare(event.message, chat)
        if pending is None:
            return
        async with self._pending_lock:
            self._pending.append(pending)
            logger.info("Message %s queued (pending=%d)", msg_id, len(self._pending))

    async def _flush_loop(self) -> None:
        while True:
            interval = max(1.0, self._config_loader.get().monitor.batch_window_minutes * 60)
            await asyncio.sleep(interval)
            await self._flush()

    async def _flush(self) -> None:
        async with self._pending_lock:
            batch, self._pending = self._pending, []
        if not batch:
            return
        config = self._config_loader.get().monitor
        await self._process_batch(config.model, config.evaluate_prompt, config.transform_prompt, batch)

    async def _backfill(self, hours: int) -> None:
        """Process messages from the last `hours` hours, oldest first.

        Runs once on startup. Relies on SeenStore for idempotency, so messages
        already handled (in a previous run or via the live stream) are skipped.
        """
        if not self._config_loader.get().monitor.enabled:
            logger.info("Backfill skipped: monitor disabled")
            return

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = []
        try:
            async for message in self._client.iter_messages(self._entity):
                if message.date < cutoff:
                    break
                recent.append(message)
        except Exception:
            logger.exception("Backfill: failed to fetch recent messages")
            return

        logger.info(
            "Backfill: found %d message(s) in the last %dh", len(recent), hours
        )
        # iter_messages yields newest→oldest; process oldest first so forwarded
        # posts keep their chronological order in the destination.
        batch: list[PendingMessage] = []
        for message in reversed(recent):
            pending = self._prepare(message, self._entity)
            if pending is not None:
                batch.append(pending)
        if batch:
            config = self._config_loader.get().monitor
            await self._process_batch(config.model, config.evaluate_prompt, config.transform_prompt, batch)
        logger.info("Backfill: complete")

    def _prepare(self, message, chat) -> PendingMessage | None:
        """Dedup + cheap pre-filtering. Returns None if the message shouldn't be
        queued for the LLM at all (already seen, empty, disabled, or keyword-skipped).

        Marks the message seen as a side effect whenever it's decided (dropped
        or queued), so restarts don't re-evaluate it.
        """
        msg_id = f"{message.chat_id}:{message.id}"
        if self._store.is_seen(DEDUP_NAMESPACE, msg_id):
            logger.info("Message %s already seen, dropping", msg_id)
            return None

        text = message.message or ""
        if not text.strip():
            # No text/caption to evaluate (pure media/sticker). Mark seen and skip.
            logger.info("Message %s has no text, dropping", msg_id)
            self._store.mark_seen(DEDUP_NAMESPACE, msg_id)
            return None

        config = self._config_loader.get().monitor
        if not config.enabled:
            logger.info("Message %s dropped: monitor disabled", msg_id)
            return None

        text = _strip_patterns(text, config.strip_patterns)
        if not text or _matches_skip_keyword(text, config.skip_keywords):
            logger.info("Message %s dropped by skip_keywords/strip_patterns", msg_id)
            self._store.mark_seen(DEDUP_NAMESPACE, msg_id)
            return None

        self._store.mark_seen(DEDUP_NAMESPACE, msg_id)
        link = self._message_link(chat, message.chat_id, message.id)
        return PendingMessage(text=text, link=link)

    async def _process_batch(
        self, model: str, evaluate_prompt: str, transform_prompt: str, pending: list[PendingMessage]
    ) -> None:
        prompt = _build_batch_prompt(evaluate_prompt, transform_prompt, pending)
        try:
            items = await self._gemini.generate_json(model, prompt)
        except Exception:
            # Messages are already marked seen (see _prepare), so a failed batch
            # is dropped rather than retried — consistent with prior behavior.
            logger.exception("Batch evaluation failed for %d message(s)", len(pending))
            return

        if not isinstance(items, list):
            logger.warning("Batch response was not a list: %r", items)
            return

        kept = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            links = [
                pending[idx].link
                for idx in (item.get("sources") or [])
                if isinstance(idx, int) and 0 <= idx < len(pending) and pending[idx].link
            ]
            await self._forward(text, links=links)
            kept += 1
        logger.info("Batch: %d message(s) in, %d forwarded", len(pending), kept)

    def _message_link(self, chat, chat_id, msg_id: int) -> str | None:
        """Build a public/private t.me link to the original message."""
        username = getattr(chat, "username", None) if chat else None
        if username:
            return f"https://t.me/{username}/{msg_id}"

        # Private channel: https://t.me/c/<internal_id>/<msg_id>
        cid = str(chat_id)
        if cid.startswith("-100"):
            return f"https://t.me/c/{cid[4:]}/{msg_id}"
        return None

    async def _forward(self, text: str, *, links: list[str] | None = None) -> None:
        body = html.escape(text)
        links = [link for link in (links or []) if link]
        if len(links) == 1:
            body = f'{body}\n\n🔗 <a href="{html.escape(links[0], quote=True)}">Original</a>'
        elif len(links) > 1:
            link_lines = "\n".join(
                f'🔗 <a href="{html.escape(link, quote=True)}">Джерело {i + 1}</a>'
                for i, link in enumerate(links)
            )
            body = f"{body}\n\n{link_lines}"
        await self._publisher.send(self._target_channel_id, body, disable_preview=True)

    async def disconnect(self) -> None:
        if self._backfill_task and not self._backfill_task.done():
            self._backfill_task.cancel()
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        try:
            await self._flush()  # don't lose a partially-filled batch on shutdown
        except Exception:
            logger.exception("Final flush on shutdown failed")
        await self._client.disconnect()
