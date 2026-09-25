"""Application entrypoint.

Runs both purposes in a single asyncio event loop:
  * APScheduler (AsyncIOScheduler) fires the daily digest.
  * Telethon's client owns the loop via run_until_disconnected().

If the daily digest fails on a rate limit, it's retried once after an hour
before alerting; any other failure (or a second rate-limited failure) alerts
the channel immediately.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import ConfigLoader
from .digest import run_digest
from .gemini import GeminiClient
from .monitor import ChannelMonitor
from .settings import Settings
from .storage import SeenStore
from .telegram_bot import Publisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("nobs")

# If the digest fails on a rate limit, wait this long and try once more before
# alerting, since free-tier per-minute throttling can clear well within an hour.
RATE_LIMIT_RETRY_SECONDS = 3600


async def _run_digest_job(
    config_loader: ConfigLoader,
    gemini: GeminiClient,
    store: SeenStore,
    publisher: Publisher,
    channel_id: str,
    *,
    is_retry: bool = False,
) -> None:
    try:
        await run_digest(
            config_loader=config_loader,
            gemini=gemini,
            store=store,
            publisher=publisher,
            channel_id=channel_id,
        )
    except Exception as exc:  # noqa: BLE001 - report any failure to the channel
        if getattr(exc, "rate_limited", False) and not is_retry:
            logger.warning(
                "Daily digest hit a rate limit; retrying once in %ds",
                RATE_LIMIT_RETRY_SECONDS,
            )
            await asyncio.sleep(RATE_LIMIT_RETRY_SECONDS)
            await _run_digest_job(
                config_loader, gemini, store, publisher, channel_id, is_retry=True
            )
            return
        logger.exception("Daily digest failed")
        try:
            await publisher.send_plain(
                channel_id, f"⚠️ Daily digest failed:\n{exc}"
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send failure alert")


async def main() -> None:
    settings = Settings.from_env()
    config_loader = ConfigLoader(settings.gcs_bucket_name, settings.config_object_path)
    gemini = GeminiClient(settings.gemini_api_key)
    store = SeenStore(settings.db_path)
    publisher = Publisher(settings.telegram_bot_token)

    # Read config once at startup to schedule the digest at the configured time.
    digest_cfg = config_loader.get(force=True).digest

    scheduler = AsyncIOScheduler(timezone=digest_cfg.timezone)
    scheduler.add_job(
        _run_digest_job,
        trigger=CronTrigger(
            hour=digest_cfg.hour,
            minute=digest_cfg.minute,
            timezone=digest_cfg.timezone,
        ),
        args=[config_loader, gemini, store, publisher, settings.telegram_channel_id],
        id="daily_digest",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info(
        "Digest scheduled daily at %02d:%02d %s",
        digest_cfg.hour,
        digest_cfg.minute,
        digest_cfg.timezone,
    )

    monitor = ChannelMonitor(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        session_string=settings.telegram_session_string,
        source_channel=settings.telegram_source_channel,
        config_loader=config_loader,
        gemini=gemini,
        store=store,
        publisher=publisher,
        target_channel_id=settings.telegram_channel_id,
    )

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass  # Windows / restricted environments

    await monitor.start()
    logger.info("NOBS is running. Press Ctrl+C to stop.")

    monitor_task = asyncio.create_task(monitor.run_until_disconnected())
    stop_task = asyncio.create_task(stop_event.wait())
    await asyncio.wait({monitor_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)

    logger.info("Shutting down…")
    scheduler.shutdown(wait=False)
    await monitor.disconnect()
    store.close()


if __name__ == "__main__":
    asyncio.run(main())
