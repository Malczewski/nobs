from __future__ import annotations

from app.config import AppConfig, ConfigLoader


def test_from_dict_parses_full_config():
    raw = {
        "digest": {
            "schedule": {"hour": 9, "minute": 30, "timezone": "Europe/Kyiv"},
            "language": "Ukrainian",
            "model": "gemini-x",
            "lookback_hours": 48,
            "summarize_prompt": "Summarize please.",
            "sources": [
                {
                    "url": "http://a.example/feed",
                    "label": "A",
                    "prompt": "Extract A.",
                    "max_items": 5,
                    "lookback_hours": 12,
                },
                {
                    "url": "http://b.example/feed",
                    "label": "B",
                    "prompt": "Extract B.",
                },
            ],
        },
        "monitor": {
            "enabled": False,
            "model": "gemini-y",
            "evaluate_prompt": "Filter.",
            "transform_prompt": "Rewrite.",
            "skip_keywords": ["skip me"],
            "strip_patterns": [r"footer$"],
            "batch_window_minutes": 2,
        },
    }

    config = AppConfig.from_dict(raw)

    assert config.digest.hour == 9
    assert config.digest.minute == 30
    assert config.digest.timezone == "Europe/Kyiv"
    assert config.digest.model == "gemini-x"
    assert len(config.digest.sources) == 2
    assert config.digest.sources[0].max_items == 5
    assert config.digest.sources[0].lookback_hours == 12
    # Falls back to the digest-level default when the source doesn't override it.
    assert config.digest.sources[1].lookback_hours == 48

    assert config.monitor.enabled is False
    assert config.monitor.model == "gemini-y"
    assert config.monitor.skip_keywords == ["skip me"]
    assert config.monitor.strip_patterns == [r"footer$"]
    assert config.monitor.batch_window_minutes == 2


def test_from_dict_applies_defaults_for_empty_config():
    config = AppConfig.from_dict({})

    assert config.digest.hour == 8
    assert config.digest.sources == []
    assert config.monitor.enabled is True
    assert config.monitor.skip_keywords == []
    assert config.monitor.strip_patterns == []
    assert config.monitor.batch_window_minutes == 5.0


class _FakeBlob:
    def __init__(self, text: str):
        self._text = text

    def download_as_text(self) -> str:
        return self._text


class _FakeStorageClient:
    """Tracks how many times a config object is actually downloaded."""

    def __init__(self, text: str):
        self._text = text
        self.bucket_calls = 0

    def bucket(self, name: str) -> "_FakeStorageClient":
        self.bucket_calls += 1
        return self

    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob(self._text)


def test_config_loader_caches_within_ttl(monkeypatch):
    fake_client = _FakeStorageClient("digest:\n  language: Ukrainian\n")
    monkeypatch.setattr("app.config.storage.Client", lambda: fake_client)

    loader = ConfigLoader("bucket", "config.yaml", ttl_seconds=1000)
    first = loader.get()
    second = loader.get()

    assert first is second
    assert first.digest.language == "Ukrainian"
    assert fake_client.bucket_calls == 1


def test_config_loader_force_refetches(monkeypatch):
    fake_client = _FakeStorageClient("digest:\n  language: Ukrainian\n")
    monkeypatch.setattr("app.config.storage.Client", lambda: fake_client)

    loader = ConfigLoader("bucket", "config.yaml", ttl_seconds=1000)
    loader.get()
    loader.get(force=True)

    assert fake_client.bucket_calls == 2


def test_config_loader_refetches_after_ttl_expires(monkeypatch):
    fake_client = _FakeStorageClient("digest:\n  language: Ukrainian\n")
    monkeypatch.setattr("app.config.storage.Client", lambda: fake_client)

    loader = ConfigLoader("bucket", "config.yaml", ttl_seconds=0)
    loader.get()
    loader.get()

    assert fake_client.bucket_calls == 2
