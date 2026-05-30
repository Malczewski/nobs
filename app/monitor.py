"""Purpose 2 — channel monitor.

A Telethon user client listens for new messages in TELEGRAM_SOURCE_CHANNEL.
Each message is evaluated by Gemini ({"keep": bool, "reason": str}); kept ones
are optionally rewritten by Gemini (monitor.transform_prompt) to strip fluff,
then immediately re-posted to the destination channel via the bot, with a link
back to the original message. Media is not re-uploaded (the link covers it).

Config (the evaluation prompt) is re-read per message — but TTL-cached in
ConfigLoader so edits in GCS apply within ~1 minute without a restart.
"""

from __future__ import annotations

import asyncio
import html
import logging
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


def _resolve_source(value: str):
    """Accept a numeric channel id, a t.me username, or a bare @username."""
    value = value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    if value.startswith("https://t.me/"):
        return value.rsplit("/", 1)[-1]
    return value.lstrip("@") if value.startswith("@") else value


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
        chat = await event.get_chat()
        await self._process_message(event.message, chat)

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
        for message in reversed(recent):
            try:
                await self._process_message(message, self._entity)
            except Exception:
                logger.exception("Backfill: failed to process message %s", message.id)
        logger.info("Backfill: complete")

    async def _process_message(self, message, chat) -> None:
        msg_id = f"{message.chat_id}:{message.id}"
        logger.debug("Processing message %s", msg_id)
        if self._store.is_seen(DEDUP_NAMESPACE, msg_id):
            return

        # For media messages, message.message holds the caption (if any).
        text = message.message or ""
        if not text.strip():
            # No text/caption to evaluate (pure media/sticker). Mark seen and skip.
            self._store.mark_seen(DEDUP_NAMESPACE, msg_id)
            return

        config = self._config_loader.get().monitor
        if not config.enabled:
            return

        try:
            verdict = await self._evaluate(config.model, config.evaluate_prompt, text)
        except Exception:
            logger.exception("Evaluation failed for message %s", msg_id)
            return  # don't mark seen → retry on next restart

        self._store.mark_seen(DEDUP_NAMESPACE, msg_id)
        keep = bool(verdict.get("keep"))
        reason = verdict.get("reason", "")
        logger.info("Message %s keep=%s reason=%s", msg_id, keep, reason)

        if not keep:
            return

        out_text = text
        if config.transform_prompt.strip():
            try:
                out_text = await self._transform(config.model, config.transform_prompt, text)
            except Exception:
                # Transform is best-effort: on failure, forward the original so
                # the message isn't lost.
                logger.exception("Transform failed for %s; forwarding original", msg_id)
                out_text = text

        link = self._message_link(chat, message.chat_id, message.id)
        await self._forward(out_text, link=link)

    async def _evaluate(self, model: str, prompt: str, text: str) -> dict:
        full_prompt = (
            f"{prompt}\n\n"
            'Respond with ONLY a JSON object: {"keep": boolean, "reason": string}.\n\n'
            f"Message:\n{text}"
        )
        result = await self._gemini.generate_json(model, full_prompt)
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object, got: {type(result)}")
        return result

    async def _transform(self, model: str, prompt: str, text: str) -> str:
        full_prompt = (
            f"{prompt}\n\n"
            "Return ONLY the rewritten message text, with no preamble, labels, "
            "or code fences.\n\n"
            f"Message:\n{text}"
        )
        result = (await self._gemini.generate(model, full_prompt)).strip()
        # Guard against an empty model response wiping the message.
        return result or text

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

    async def _forward(self, text: str, *, link: str | None = None) -> None:
        body = html.escape(text)
        if link:
            body = f'{body}\n\n🔗 <a href="{html.escape(link, quote=True)}">Original</a>'
        await self._publisher.send(self._target_channel_id, body, disable_preview=True)

    async def disconnect(self) -> None:
        if self._backfill_task and not self._backfill_task.done():
            self._backfill_task.cancel()
        await self._client.disconnect()
