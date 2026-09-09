"""Offline provider tests: no OpenAI installation, credentials or network needed."""

import sys
from types import SimpleNamespace

from pydantic import BaseModel
import pytest

from app.providers.base import ModelProvider
from app.providers.openai import OpenAIProvider, OpenAIProviderError


class Verdict(BaseModel):
    verdict: str


class FakeResponses:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def _call(self, method, **kwargs):
        self.calls.append((method, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def parse(self, **kwargs):
        return self._call("parse", **kwargs)

    def create(self, **kwargs):
        return self._call("create", **kwargs)


def response(**overrides):
    values = {
        "status": "completed",
        "output": [],
        "output_text": "Ready",
        "output_parsed": Verdict(verdict="APPLY"),
        "usage": SimpleNamespace(input_tokens=120, output_tokens=30),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def provider_for(*responses, **kwargs):
    client = SimpleNamespace(responses=FakeResponses(*responses))
    return OpenAIProvider(model="configured-test-model", client=client, **kwargs)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # Never load the developer's .env during tests.
    monkeypatch.setattr("app.providers.openai._load_environment", lambda: None)
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(name, raising=False)


def test_structured_output_passes_schema_and_messages_and_disables_storage():
    provider = provider_for(response(), max_output_tokens=2200)

    actual = provider.generate_structured("Candidate facts", Verdict, "Extract facts.")

    assert actual == Verdict(verdict="APPLY")
    assert isinstance(provider, ModelProvider)
    assert provider.client.responses.calls == [("parse", {
        "model": "configured-test-model",
        "input": [
            {"role": "system", "content": "Extract facts."},
            {"role": "user", "content": "Candidate facts"},
        ],
        "max_output_tokens": 2200,
        "store": False,
        "text_format": Verdict,
    })]


def test_generate_uses_text_response_and_usage_accumulates():
    provider = provider_for(response(), response())

    assert provider.generate("Text prompt") == "Ready"
    provider.generate_structured("Structured prompt", Verdict)

    assert provider.client.responses.calls[0][0] == "create"
    assert provider.client.responses.calls[0][1]["store"] is False
    assert provider.usage == {
        "requests": 2, "input_tokens": 240, "output_tokens": 60, "total_tokens": 300,
    }
    provider.usage["requests"] = 900
    assert provider.usage["requests"] == 2


@pytest.mark.parametrize("status", ["failed", "queued", "in_progress", "cancelled", None])
def test_only_completed_results_are_usable(status):
    provider = provider_for(response(status=status))
    with pytest.raises(OpenAIProviderError, match="completed response"):
        provider.generate_structured("prompt", Verdict)
    assert provider.usage["input_tokens"] == 120


def test_incomplete_response_explains_output_limit_and_preserves_usage():
    provider = provider_for(response(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    ))
    with pytest.raises(OpenAIProviderError, match="Increase OPENAI_MAX_OUTPUT_TOKENS"):
        provider.generate_structured("prompt", Verdict)
    assert provider.usage["total_tokens"] == 150


def test_refusal_is_rejected_even_with_a_parsed_result():
    provider = provider_for(response(output=[SimpleNamespace(content=[
        SimpleNamespace(type="refusal", refusal="Private refusal text"),
    ])]))
    with pytest.raises(OpenAIProviderError, match="refused") as error:
        provider.generate_structured("prompt", Verdict)
    assert "Private refusal text" not in str(error.value)


def test_missing_structured_output_is_rejected():
    provider = provider_for(response(output_parsed=None))
    with pytest.raises(OpenAIProviderError, match="no parsed structured output"):
        provider.generate_structured("prompt", Verdict)


def test_invalid_parsed_data_is_rejected_without_exposing_content():
    provider = provider_for(response(output_parsed={"private": "candidate data"}))
    with pytest.raises(OpenAIProviderError, match="schema validation") as error:
        provider.generate_structured("prompt", Verdict)
    assert "candidate data" not in str(error.value)


@pytest.mark.parametrize("text", [None, "", "   "])
def test_missing_text_is_rejected(text):
    provider = provider_for(response(output_text=text))
    with pytest.raises(OpenAIProviderError, match="no text output"):
        provider.generate("prompt")


@pytest.mark.parametrize("status,advice", [(401, "OPENAI_API_KEY"), (429, "quota/rate"), (400, "OPENAI_MODEL")])
def test_api_errors_are_actionable_and_do_not_expose_secrets(status, advice):
    exception = RuntimeError("A secret API key or CV was in this error")
    exception.status_code = status
    provider = provider_for(exception)
    with pytest.raises(OpenAIProviderError, match=advice) as error:
        provider.generate("prompt")
    assert "secret" not in str(error.value)
    assert error.value.__suppress_context__ is True
    assert provider.usage == {
        "requests": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    }


def test_model_is_required_even_with_an_injected_client():
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        OpenAIProvider(client=object())


def test_missing_key_is_reported_before_sdk_import():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider(model="test-model")


def test_client_is_built_with_env_config_and_bounded_retry_settings(monkeypatch):
    calls = []
    fake_client = object()

    def make_client(**kwargs):
        calls.append(kwargs)
        return fake_client

    monkeypatch.setenv("OPENAI_MODEL", "chosen-model")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "3100")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=make_client))

    provider = OpenAIProvider(timeout=22.0, max_retries=1)

    assert provider.model == "chosen-model"
    assert provider.max_output_tokens == 3100
    assert provider.client is fake_client
    assert calls == [{"api_key": "dummy-test-key", "timeout": 22.0, "max_retries": 1}]


def test_sdk_is_optional_and_missing_install_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="pip install"):
        OpenAIProvider(model="test-model", api_key="dummy-test-key")


@pytest.mark.parametrize("kwargs", [
    {"timeout": 0}, {"timeout": 301}, {"timeout": float("nan")},
    {"max_retries": -1}, {"max_retries": 6}, {"max_retries": True},
    {"max_output_tokens": 0}, {"max_output_tokens": 128001}, {"max_output_tokens": True},
])
def test_unbounded_or_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        provider_for(**kwargs)


def test_invalid_env_token_limit_has_actionable_error(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "unbounded")
    with pytest.raises(ValueError, match="positive integer"):
        provider_for()


def test_invalid_schema_does_not_make_a_request():
    provider = provider_for()
    with pytest.raises(TypeError, match="BaseModel"):
        provider.generate_structured("prompt", dict)
    assert provider.usage["requests"] == 0
