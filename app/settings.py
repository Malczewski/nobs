"""Environment-driven settings.

Everything that is *secret* or *deployment-specific* lives in environment
variables. Everything that is *editable content* (prompts, feed URLs, schedule)
lives in the GCS `config.yaml` so it can change without a redeploy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(RuntimeError):
    """Raised when a required environment variable is missing."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SettingsError(f"Missing required environment variable: {name}")
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # Telegram bot (posting/forwarding).
    telegram_bot_token: str
    telegram_channel_id: str  # destination for digests AND forwarded posts

    # Telegram user account (Telethon / MTProto) for reading channels.
    telegram_api_id: int
    telegram_api_hash: str
    telegram_phone: str
    telegram_session_string: str
    telegram_source_channel: str  # channel that the monitor watches

    # Gemini.
    gemini_api_key: str

    # GCS config.
    gcs_bucket_name: str
    config_object_path: str

    # Local state.
    db_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        api_id_raw = _require("TELEGRAM_API_ID")
        try:
            api_id = int(api_id_raw)
        except ValueError as exc:
            raise SettingsError("TELEGRAM_API_ID must be an integer") from exc

        return cls(
            telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
            telegram_channel_id=_require("TELEGRAM_CHANNEL_ID"),
            telegram_api_id=api_id,
            telegram_api_hash=_require("TELEGRAM_API_HASH"),
            telegram_phone=_optional("TELEGRAM_PHONE"),
            telegram_session_string=_require("TELEGRAM_SESSION_STRING"),
            telegram_source_channel=_require("TELEGRAM_SOURCE_CHANNEL"),
            gemini_api_key=_require("GEMINI_API_KEY"),
            gcs_bucket_name=_require("GCS_BUCKET_NAME"),
            config_object_path=_optional("GCS_CONFIG_PATH", "config.yaml"),
            db_path=_optional("DB_PATH", "/data/nobs.db"),
        )
