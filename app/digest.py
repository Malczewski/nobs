"""Purpose 1 — daily news digest.

For each configured source: fetch the feed, drop entries we've already seen,
ask Gemini to distill + translate the remaining items into a compact JSON
array, then render and post one message per source to the channel.
"""

from __future__ import annotations

import asyncio
import calendar
import html
import logging
import time
from dataclasses import dataclass

import feedparser

from .config import ConfigLoader, Source
from .gemini import GeminiClient
from .storage import SeenStore
from .telegram_bot import Publisher

logger = logging.getLogger(__name__)

DEDUP_NAMESPACE = "rss"


@dataclass
class FeedEntry:
    entry_id: str
    title: str
    link: str
    summary: str


def _entry_id(entry: dict) -> str:
    return (
        entry.get("id")
        or entry.get("guid")
        or entry.get("link")
        or entry.get("title", "")
    )


def _entry_epoch(entry: dict) -> float | None:
    """Best-effort UTC publish time (epoch seconds) from a feed entry."""
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        struct = entry.get(key)
        if struct:
            # feedparser returns time.struct_time already normalized to UTC.
            return calendar.timegm(struct)
    return None


def fetch_feed(source: Source) -> list[FeedEntry]:
    parsed = feedparser.parse(source.url)
    if parsed.bozo and not parsed.entries:
        logger.warning("Feed parse issue for %s: %s", source.url, parsed.bozo_exception)

    cutoff = time.time() - source.lookback_hours * 3600
    entries: list[FeedEntry] = []
    skipped_old = 0
    for raw in parsed.entries:
        published = _entry_epoch(raw)
        # Keep entries published within the window. Entries without a usable date
        # are kept (SQLite dedup prevents reposting them on later runs).
        if published is not None and published < cutoff:
            skipped_old += 1
            continue
        entries.append(
            FeedEntry(
                entry_id=_entry_id(raw),
                title=(raw.get("title") or "").strip(),
                link=(raw.get("link") or "").strip(),
                summary=(raw.get("summary") or raw.get("description") or "").strip(),
            )
        )
        if len(entries) >= source.max_items:
            break

    logger.info(
        "[%s] %d entries within %dh (skipped %d older)",
        source.label,
        len(entries),
        source.lookback_hours,
        skipped_old,
    )
    return entries


def _build_prompt(source: Source, language: str, summarize_prompt: str, entries: list[FeedEntry]) -> str:
    items_block = "\n\n".join(
        f"- title: {e.title}\n  link: {e.link}\n  description: {e.summary}"
        for e in entries
    )
    return (
        f"{source.prompt}\n\n"
        f"{summarize_prompt}\n\n"
        f"Translate every title and summary into {language}. Keep proper nouns, "
        f"product names and links unchanged. Use the exact 'link' value provided "
        f"for each item; never invent or modify links. If nothing qualifies, "
        f"return an empty JSON array [].\n\n"
        f"News:\n{items_block}"
    )


def _render_message(label: str, items: list[dict]) -> str:
    """Render to the requested format using safe HTML.

    {label}:
    * {title} (link)
    {summary}
    """
    lines = [f"<b>{html.escape(label)}</b>"]
    for item in items:
        title = html.escape(str(item.get("title", "")).strip())
        link = str(item.get("link", "")).strip()
        summary = html.escape(str(item.get("summary", "")).strip())
        if link:
            lines.append(f'• <a href="{html.escape(link, quote=True)}">{title}</a>')
        else:
            lines.append(f"• {title}")
        if summary:
            lines.append(summary)
        lines.append("")
    return "\n".join(lines).strip()


async def process_source(
    source: Source,
    *,
    language: str,
    model: str,
    summarize_prompt: str,
    gemini: GeminiClient,
    store: SeenStore,
) -> str | None:
    entries = await asyncio.to_thread(fetch_feed, source)
    fresh = [e for e in entries if not store.is_seen(DEDUP_NAMESPACE, e.entry_id)]
    if not fresh:
        logger.info("[%s] no fresh entries", source.label)
        return None

    prompt = _build_prompt(source, language, summarize_prompt, fresh)
    try:
        items = await gemini.generate_json(model, prompt)
    except Exception:
        logger.exception("[%s] Gemini summarization failed", source.label)
        raise

    # Mark all fetched entries as seen regardless of whether the model kept them,
    # so we don't reconsider the same items tomorrow.
    store.mark_seen_many(DEDUP_NAMESPACE, [e.entry_id for e in fresh])

    if not isinstance(items, list) or not items:
        logger.info("[%s] model returned nothing notable", source.label)
        return None

    return _render_message(source.label, items)


async def run_digest(
    *,
    config_loader: ConfigLoader,
    gemini: GeminiClient,
    store: SeenStore,
    publisher: Publisher,
    channel_id: str,
) -> None:
    """Run a full digest pass. Re-reads config from GCS first."""
    config = config_loader.get(force=True).digest
    logger.info("Running digest across %d sources", len(config.sources))

    errors: list[str] = []
    posted = 0
    for source in config.sources:
        try:
            message = await process_source(
                source,
                language=config.language,
                model=config.model,
                summarize_prompt=config.summarize_prompt,
                gemini=gemini,
                store=store,
            )
            if message:
                await publisher.send(channel_id, message, disable_preview=True)
                posted += 1
        except Exception as exc:  # noqa: BLE001 - collect & report per source
            logger.exception("Source failed: %s", source.label)
            errors.append(f"{source.label or source.url}: {exc}")

    logger.info("Digest complete: %d messages posted, %d errors", posted, len(errors))
    if errors:
        # Surface partial failures but don't crash the whole run.
        raise DigestPartialFailure(errors)


class DigestPartialFailure(RuntimeError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))
