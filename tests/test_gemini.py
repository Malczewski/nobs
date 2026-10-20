from __future__ import annotations

import json

import pytest
from google.api_core import exceptions as gexc

from app import gemini as gm


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_is_retryable_true_for_transient_errors():
    assert gm._is_retryable(gexc.ServiceUnavailable("down")) is True
    assert gm._is_retryable(gexc.ResourceExhausted("rate limited, retry soon")) is True


def test_is_retryable_false_for_daily_quota():
    exc = gexc.ResourceExhausted(
        "quota_id: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    )
    assert gm._is_retryable(exc) is False


def test_is_retryable_false_for_depleted_prepayment_credits():
    exc = gexc.ResourceExhausted(
        "Your prepayment credits are depleted. Please go to AI Studio at "
        "https://ai.studio/projects to manage your project and billing."
    )
    assert gm._is_retryable(exc) is False


def test_is_retryable_false_for_non_retryable_exception_types():
    assert gm._is_retryable(ValueError("not a gemini error")) is False


def test_retry_after_seconds_parses_textual_retry_delay():
    exc = Exception("blah retry_delay { seconds: 48 } blah")
    assert gm._retry_after_seconds(exc) == 48.0


def test_retry_after_seconds_none_when_absent():
    assert gm._retry_after_seconds(Exception("no hint here")) is None


def test_retry_after_seconds_none_for_none_input():
    assert gm._retry_after_seconds(None) is None


def test_parse_json_plain():
    assert gm._parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_code_fence():
    text = '```json\n{"a": 1}\n```'
    assert gm._parse_json(text) == {"a": 1}


def test_parse_json_strips_bare_fence():
    text = "```\n[1, 2, 3]\n```"
    assert gm._parse_json(text) == [1, 2, 3]


def test_parse_json_extracts_embedded_object():
    text = 'Sure, here you go:\n{"a": 1}\nHope that helps!'
    assert gm._parse_json(text) == {"a": 1}


def test_parse_json_raises_when_nothing_parseable():
    with pytest.raises(json.JSONDecodeError):
        gm._parse_json("not json at all")


# ---------------------------------------------------------------------------
# GeminiClient integration (genai + tenacity wired together, no real network)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    """Shrink backoff/throttle constants so retry tests run in milliseconds."""
    monkeypatch.setattr(gm, "_DEFAULT_RETRY_INITIAL", 0.001)
    monkeypatch.setattr(gm, "_DEFAULT_RETRY_MAX", 0.002)
    monkeypatch.setattr(gm, "_MIN_REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr(gm.genai, "configure", lambda **kwargs: None)


class _FakeModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_content(self, prompt):
        self.calls += 1
        result = self._responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return _FakeResponse(result)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


async def test_generate_succeeds_first_try(monkeypatch):
    model = _FakeModel(["hello"])
    monkeypatch.setattr(gm.genai, "GenerativeModel", lambda name: model)

    client = gm.GeminiClient("fake-key")
    result = await client.generate("model", "prompt")

    assert result == "hello"
    assert model.calls == 1


async def test_generate_retries_transient_error_then_succeeds(monkeypatch):
    model = _FakeModel([gexc.ServiceUnavailable("down"), "recovered"])
    monkeypatch.setattr(gm.genai, "GenerativeModel", lambda name: model)

    client = gm.GeminiClient("fake-key")
    result = await client.generate("model", "prompt")

    assert result == "recovered"
    assert model.calls == 2


async def test_generate_fails_fast_on_daily_quota(monkeypatch):
    daily_quota_error = gexc.ResourceExhausted(
        "quota_id: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    )
    model = _FakeModel([daily_quota_error, "should never be reached"])
    monkeypatch.setattr(gm.genai, "GenerativeModel", lambda name: model)

    client = gm.GeminiClient("fake-key")
    with pytest.raises(gexc.ResourceExhausted):
        await client.generate("model", "prompt")

    assert model.calls == 1  # no retry burned on a wait that can't help


async def test_generate_exhausts_retries_and_raises(monkeypatch):
    model = _FakeModel([gexc.ServiceUnavailable("down")] * gm._RETRY_ATTEMPTS)
    monkeypatch.setattr(gm.genai, "GenerativeModel", lambda name: model)

    client = gm.GeminiClient("fake-key")
    with pytest.raises(gexc.ServiceUnavailable):
        await client.generate("model", "prompt")

    assert model.calls == gm._RETRY_ATTEMPTS


async def test_generate_json_parses_response(monkeypatch):
    model = _FakeModel(['```json\n[{"title": "x"}]\n```'])
    monkeypatch.setattr(gm.genai, "GenerativeModel", lambda name: model)

    client = gm.GeminiClient("fake-key")
    result = await client.generate_json("model", "prompt")

    assert result == [{"title": "x"}]


# ---------------------------------------------------------------------------
# Friendly quota errors + local usage counting
# ---------------------------------------------------------------------------

_DAILY_429 = (
    "429 You exceeded your current quota. * Quota exceeded for metric: "
    "generate_content_free_tier_requests, limit: 500, model: gemini-x "
    'violations { quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier" '
    "quota_value: 500 } retry_delay { seconds: 2631 }"
)


async def test_daily_quota_raises_readable_error(monkeypatch):
    model = _FakeModel([gexc.ResourceExhausted(_DAILY_429)])
    monkeypatch.setattr(gm.genai, "GenerativeModel", lambda name: model)

    client = gm.GeminiClient("fake-key")
    with pytest.raises(gm.GeminiQuotaError) as exc_info:
        await client.generate("gemini-x", "prompt")

    message = str(exc_info.value)
    assert exc_info.value.kind == "per_day"
    assert "daily request quota" in message and "500/day" in message
    assert "violations" not in message and "retry_delay" not in message
    assert "1 request(s) today" in message  # failed attempts are counted too


def test_describe_error_handles_raw_quota_and_other_errors():
    assert "daily request quota" in gm.describe_error(gexc.ResourceExhausted(_DAILY_429))
    assert gm.describe_error(ValueError("bad\nthing")) == "ValueError: bad thing"


async def test_successful_request_is_logged_with_local_count(monkeypatch, caplog):
    model = _FakeModel(["a", "b"])
    monkeypatch.setattr(gm.genai, "GenerativeModel", lambda name: model)
    client = gm.GeminiClient("fake-key")

    with caplog.at_level("INFO", logger="app.gemini"):
        await client.generate("gemini-x", "prompt", label="t")
        await client.generate("gemini-x", "prompt", label="t")

    assert "label=t" in caplog.text
    assert "local_today=2" in caplog.text
