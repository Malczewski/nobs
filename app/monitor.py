"""Purpose 2 — channel monitor.

A Telethon user client listens for new messages in TELEGRAM_SOURCE_CHANNEL.
Each message is evaluated by Gemini ({"keep": bool, "reason": str}); kept ones
are immediately re-posted to the destination channel via the bot.

Config (the evaluation prompt) is re-read per message — but TTL-cached in
ConfigLoader so edits in GCS apply within ~1 minute without a restart.
"""

from __future__ import annotations

import logging

from telethon import TelegramClient, events
from telethon.sessions import StringSession

from .config import ConfigLoader
from .gemini import GeminiClient
from .storage import SeenStore
from .telegram_bot import Publisher

logger = logging.getLogger(__name__)

DEDUP_NAMESPACE = "tg_monitor"


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

    async def start(self) -> None:
        await self._client.start()  # uses the StringSession; non-interactive
        entity = await self._client.get_entity(self._source_channel)
        self._client.add_event_handler(
            self._on_new_message, events.NewMessage(chats=entity)
        )
        logger.info("Monitor listening on source channel: %s", self._source_channel)

    async def run_forever(self) -> None:
        await self.start()
        await self._client.run_until_disconnected()

    def run_until_disconnected(self):
        """Awaitable that resolves when the underlying client disconnects."""
        return self._client.run_until_disconnected()

    async def _on_new_message(self, event: events.NewMessage.Event) -> None:
        message = event.message
        msg_id = f"{event.chat_id}:{message.id}"
        if self._store.is_seen(DEDUP_NAMESPACE, msg_id):
            return

        text = message.message or ""
        if not text.strip():
            # Nothing to evaluate (pure media/sticker). Mark seen and skip.
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

        if keep:
            await self._forward(text)

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

    async def _forward(self, text: str) -> None:
        await self._publisher.send_plain(self._target_channel_id, text)

    async def disconnect(self) -> None:
        await self._client.disconnect()
