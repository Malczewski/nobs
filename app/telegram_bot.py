"""Posting/forwarding via python-telegram-bot.

A single Bot instance is shared by the digest publisher and the monitor
forwarder. Long digests are split to respect Telegram's 4096-char limit.
"""

from __future__ import annotations

import logging

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut
from telegram.request import HTTPXRequest

logger = logging.getLogger(__name__)

TELEGRAM_MAX_CHARS = 4096


class Publisher:
    def __init__(self, bot_token: str):
        # Slightly larger pool/timeouts: digests can be a burst of messages.
        request = HTTPXRequest(connection_pool_size=8, read_timeout=30, write_timeout=30)
        self._bot = Bot(token=bot_token, request=request)

    async def send(self, chat_id: str, text: str, *, disable_preview: bool = False) -> None:
        for chunk in _split_message(text):
            await self._send_one(chat_id, chunk, disable_preview=disable_preview)

    async def _send_one(self, chat_id: str, text: str, *, disable_preview: bool) -> None:
        try:
            await self._bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=disable_preview,
            )
        except RetryAfter as exc:
            import asyncio

            logger.warning("Telegram rate limited; sleeping %ss", exc.retry_after)
            await asyncio.sleep(exc.retry_after + 1)
            await self._bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=disable_preview,
            )
        except TimedOut:
            logger.warning("Telegram send timed out; retrying once")
            await self._bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=disable_preview,
            )

    async def send_plain(self, chat_id: str, text: str) -> None:
        """Send without HTML parsing (used for failure alerts / arbitrary text)."""
        for chunk in _split_message(text):
            await self._bot.send_message(chat_id=chat_id, text=chunk)


def _split_message(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        # +1 for the newline we re-add.
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            # A single line longer than the limit gets hard-split.
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks
