"""Runtime configuration loaded from a `config.yaml` object in GCS.

The config holds *editable content*: the digest schedule, per-source feed URLs
and prompts, and the monitor's evaluation prompt. It is re-read from GCS so that
edits take effect without a restart or redeploy.

To avoid hammering GCS on every single incoming message (the monitor can be
chatty), reads are cached for a short TTL. A digest run or a message batch will
always observe edits made more than `ttl_seconds` ago — which in practice means
"edit in GCS, wait up to a minute, done". Set CONFIG_TTL_SECONDS=0 to always
fetch fresh.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import yaml
from google.cloud import storage

logger = logging.getLogger(__name__)


@dataclass
class Source:
    url: str
    label: str
    prompt: str
    max_items: int = 30  # hard cap on entries sent to Gemini (safety)
    lookback_hours: int = 24  # only consider entries published within this window


@dataclass
class DigestConfig:
    hour: int = 8
    minute: int = 0
    timezone: str = "Europe/Kyiv"
    language: str = "Ukrainian"
    model: str = "gemini-3.5-flash"
    summarize_prompt: str = ""
    sources: list[Source] = field(default_factory=list)


@dataclass
class MonitorConfig:
    enabled: bool = True
    model: str = "gemini-3.5-flash"
    evaluate_prompt: str = ""
    # Optional: rewrite kept messages into a more useful form before forwarding.
    # If empty, the original message text is forwarded unchanged.
    transform_prompt: str = ""


@dataclass
class AppConfig:
    digest: DigestConfig
    monitor: MonitorConfig

    @classmethod
    def from_dict(cls, raw: dict) -> "AppConfig":
        digest_raw = raw.get("digest", {}) or {}
        sched = digest_raw.get("schedule", {}) or {}
        default_lookback = int(digest_raw.get("lookback_hours", 24))
        sources = [
            Source(
                url=s["url"],
                label=s.get("label", ""),
                prompt=s.get("prompt", ""),
                max_items=int(s.get("max_items", 30)),
                lookback_hours=int(s.get("lookback_hours", default_lookback)),
            )
            for s in (digest_raw.get("sources", []) or [])
        ]
        digest = DigestConfig(
            hour=int(sched.get("hour", 8)),
            minute=int(sched.get("minute", 0)),
            timezone=str(sched.get("timezone", "Europe/Kyiv")),
            language=str(digest_raw.get("language", "Ukrainian")),
            model=str(digest_raw.get("model", "gemini-3.5-flash")),
            summarize_prompt=str(digest_raw.get("summarize_prompt", "")),
            sources=sources,
        )

        monitor_raw = raw.get("monitor", {}) or {}
        monitor = MonitorConfig(
            enabled=bool(monitor_raw.get("enabled", True)),
            model=str(monitor_raw.get("model", "gemini-3.5-flash")),
            evaluate_prompt=str(monitor_raw.get("evaluate_prompt", "")),
            transform_prompt=str(monitor_raw.get("transform_prompt", "")),
        )
        return cls(digest=digest, monitor=monitor)


class ConfigLoader:
    """Thread-safe, TTL-cached loader for the GCS config object."""

    def __init__(self, bucket_name: str, object_path: str, ttl_seconds: Optional[int] = None):
        self._bucket_name = bucket_name
        self._object_path = object_path
        self._ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else int(os.environ.get("CONFIG_TTL_SECONDS", "60"))
        )
        self._client = storage.Client()
        self._lock = threading.Lock()
        self._cached: Optional[AppConfig] = None
        self._fetched_at: float = 0.0

    def _fetch(self) -> AppConfig:
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(self._object_path)
        text = blob.download_as_text()
        raw = yaml.safe_load(text) or {}
        logger.info(
            "Loaded config from gs://%s/%s", self._bucket_name, self._object_path
        )
        return AppConfig.from_dict(raw)

    def get(self, *, force: bool = False) -> AppConfig:
        now = time.monotonic()
        with self._lock:
            fresh_enough = (
                self._cached is not None and (now - self._fetched_at) < self._ttl
            )
            if not force and fresh_enough:
                return self._cached
            config = self._fetch()
            self._cached = config
            self._fetched_at = now
            return config
