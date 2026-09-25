from __future__ import annotations

import pytest

from app.settings import Settings, SettingsError

REQUIRED_VARS = {
    "TELEGRAM_BOT_TOKEN": "bot-token",
    "TELEGRAM_CHANNEL_ID": "chan-id",
    "TELEGRAM_API_ID": "12345",
    "TELEGRAM_API_HASH": "api-hash",
    "TELEGRAM_SESSION_STRING": "session-string",
    "TELEGRAM_SOURCE_CHANNEL": "source-channel",
    "GEMINI_API_KEY": "gemini-key",
    "GCS_BUCKET_NAME": "bucket",
}

ALL_SETTINGS_VARS = REQUIRED_VARS | {
    "TELEGRAM_PHONE": "",
    "GCS_CONFIG_PATH": "",
    "DB_PATH": "",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate Settings.from_env() from whatever is set in the real shell/.env."""
    for name in ALL_SETTINGS_VARS:
        monkeypatch.delenv(name, raising=False)


def _set_all_required(monkeypatch):
    for name, value in REQUIRED_VARS.items():
        monkeypatch.setenv(name, value)


def test_from_env_builds_settings_with_required_vars(monkeypatch):
    _set_all_required(monkeypatch)

    settings = Settings.from_env()

    assert settings.telegram_bot_token == "bot-token"
    assert settings.telegram_api_id == 12345
    assert settings.gemini_api_key == "gemini-key"
    assert settings.config_object_path == "config.yaml"  # default
    assert settings.db_path == "/data/nobs.db"  # default


def test_from_env_uses_optional_overrides(monkeypatch):
    _set_all_required(monkeypatch)
    monkeypatch.setenv("GCS_CONFIG_PATH", "custom.yaml")
    monkeypatch.setenv("DB_PATH", "/tmp/custom.db")
    monkeypatch.setenv("TELEGRAM_PHONE", "+1000")

    settings = Settings.from_env()

    assert settings.config_object_path == "custom.yaml"
    assert settings.db_path == "/tmp/custom.db"
    assert settings.telegram_phone == "+1000"


@pytest.mark.parametrize("missing", sorted(REQUIRED_VARS))
def test_from_env_raises_when_a_required_var_is_missing(monkeypatch, missing):
    _set_all_required(monkeypatch)
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(SettingsError):
        Settings.from_env()


def test_from_env_raises_on_non_integer_api_id(monkeypatch):
    _set_all_required(monkeypatch)
    monkeypatch.setenv("TELEGRAM_API_ID", "not-a-number")

    with pytest.raises(SettingsError):
        Settings.from_env()
