"""Local-provider behavior with fake responses only; never starts inference."""

import json
import sys
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict
import pytest

from app.providers.base import ModelProvider, ModelProviderError
from app.providers.ollama_provider import OllamaProvider, OllamaProviderError


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: str


class FakeClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def response(**changes):
    payload = {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": '{"verdict":"APPLY"}'}, "prompt_eval_count": 100, "eval_count": 20}
    payload.update(changes)
    return payload


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch):
    monkeypatch.setattr("app.providers.ollama_provider._load_environment", lambda: None)
    for name in ("OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_TIMEOUT_SECONDS", "OLLAMA_NUM_CTX", "OLLAMA_NUM_PREDICT"):
        monkeypatch.delenv(name, raising=False)


def test_defaults_and_structured_api_contract_preserve_user_content():
    client = FakeClient(response())
    provider = OllamaProvider(client=client)
    result = provider.generate_structured("Complete CV and job content", Verdict, "Use only source facts.")
    assert result == Verdict(verdict="APPLY")
    assert isinstance(provider, ModelProvider)
    assert provider.name == "ollama"
    call = client.calls[0]
    assert call["model"] == "qwen3:8b"
    assert call["stream"] is False
    assert call["think"] is False
    assert call["format"] == Verdict.model_json_schema()
    assert call["options"] == {"temperature": 0, "num_ctx": 32768, "num_predict": 4096}
    assert call["messages"][1] == {"role": "user", "content": "Complete CV and job content"}
    assert call["messages"][0]["content"].startswith("Use only source facts.")
    assert '"verdict"' in call["messages"][0]["content"]


def test_generate_returns_final_answer_and_tracks_usage_not_thinking():
    client = FakeClient(response(message={"content": "Answer", "thinking": "Private reasoning"}), response())
    provider = OllamaProvider("qwen3:8b", "http://localhost:11434", client=client)
    assert provider.generate("Prompt") == "Answer"
    provider.generate_structured("Prompt", Verdict)
    assert "format" not in client.calls[0]
    assert provider.usage == {"requests": 2, "input_tokens": 200, "output_tokens": 40, "total_tokens": 240}
    provider.usage["requests"] = 99
    assert provider.usage["requests"] == 2


def test_gemma_single_user_instructions_preserve_schema_and_omit_think():
    client = FakeClient(response())
    provider = OllamaProvider(model="gemma3:4b", client=client,
                              system_message_mode="merge_user", think=None)
    assert provider.generate_structured("Complete source", Verdict, "Critique facts only.").verdict == "APPLY"
    call = client.calls[0]
    assert "think" not in call
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"
    content = call["messages"][0]["content"]
    assert content.startswith("Critique facts only.")
    assert '"verdict"' in content
    assert content.endswith("\n\nComplete source")
    assert call["format"] == Verdict.model_json_schema()


@pytest.mark.parametrize("kwargs", [{"system_message_mode": "invalid"}, {"think": "false"}, {"think": 1}])
def test_invalid_chat_capabilities_rejected(kwargs):
    with pytest.raises(ValueError):
        OllamaProvider(client=FakeClient(), **kwargs)


def test_explicit_cached_counter_preserves_full_prompt_and_reserves_output():
    observed = []
    def count(messages):
        observed.append(messages)
        return 100
    client = FakeClient(response(prompt_eval_count=130))
    provider = OllamaProvider(client=client, num_ctx=1024, num_predict=256,
                              input_token_counter=count, keep_alive=0)
    prompt = "Complete source content. " * 200
    provider.generate(prompt)
    assert client.calls[0]["messages"][0]["content"] == prompt
    assert observed == [client.calls[0]["messages"]]
    assert client.calls[0]["keep_alive"] == 0
    assert provider.budget_diagnostics[0]["estimated_input_with_reserve"] == 612
    assert provider.budget_diagnostics[0]["reported_prompt_tokens"] == 130


@pytest.mark.parametrize("count", [None, True, -1, 0, 1.5, "100"])
def test_invalid_cached_counter_result_never_sends_request(count):
    client = FakeClient()
    provider = OllamaProvider(client=client, input_token_counter=lambda _: count)
    with pytest.raises(OllamaProviderError, match="invalid prompt count"):
        provider.generate("Complete source")
    assert not client.calls


def test_cached_count_still_enforces_complete_context_budget():
    client = FakeClient()
    provider = OllamaProvider(client=client, num_ctx=1024, num_predict=256,
                              input_token_counter=lambda _: 400)
    with pytest.raises(OllamaProviderError, match="context allowance"):
        provider.generate("Complete source")
    assert not client.calls


@pytest.mark.parametrize("reported", [0, -1, True, None, 100.0, "100", 613])
def test_cached_counter_requires_reconciled_ollama_prompt_accounting(reported):
    provider = OllamaProvider(client=FakeClient(response(prompt_eval_count=reported)),
                              input_token_counter=lambda _: 100)
    with pytest.raises(OllamaProviderError, match="reconciled"):
        provider.generate("Complete source")


@pytest.mark.parametrize("structured,reported,tolerance", [(False, 96, 4), (True, 92, 8)])
def test_cached_counter_allows_small_content_framing_boundary_offset(structured, reported, tolerance):
    provider = OllamaProvider(client=FakeClient(response(prompt_eval_count=reported)),
                              input_token_counter=lambda _: 100)
    if structured:
        assert provider.generate_structured("Complete source", Verdict) == Verdict(verdict="APPLY")
    else:
        assert provider.generate("Complete source") == '{"verdict":"APPLY"}'
    diagnostics = provider.budget_diagnostics[-1]
    assert diagnostics["counted_text_tokens"] == 100
    assert diagnostics["boundary_tolerance_tokens"] == tolerance
    assert diagnostics["minimum_reported_prompt_tokens"] == 100 - tolerance


@pytest.mark.parametrize("structured,reported", [(False, 95), (True, 91)])
def test_cached_counter_rejects_undercount_beyond_small_boundary_tolerance(structured, reported):
    provider = OllamaProvider(client=FakeClient(response(prompt_eval_count=reported)),
                              input_token_counter=lambda _: 100)
    with pytest.raises(OllamaProviderError, match="reconciled"):
        if structured:
            provider.generate_structured("Complete source", Verdict)
        else:
            provider.generate("Complete source")


def test_cached_counter_rejects_severe_undercount_and_retains_original_text_count():
    provider = OllamaProvider(client=FakeClient(response(prompt_eval_count=100, eval_count=20)),
                              input_token_counter=lambda _: 6000)
    with pytest.raises(OllamaProviderError, match="reconciled"):
        provider.generate("Complete source")
    # Usage processing must not overwrite the preflight count used for this check.
    assert provider.budget_diagnostics[-1]["counted_text_tokens"] == 6000
    assert provider.budget_diagnostics[-1]["minimum_reported_prompt_tokens"] == 5996
    assert provider.usage["input_tokens"] == 100


def test_cached_counter_accepts_upper_reserve_boundary():
    provider = OllamaProvider(client=FakeClient(response(prompt_eval_count=612)),
                              input_token_counter=lambda _: 100)
    assert provider.generate("Complete source") == '{"verdict":"APPLY"}'


def test_default_byte_estimate_is_not_used_as_a_lower_token_bound():
    provider = OllamaProvider(client=FakeClient(response(prompt_eval_count=1)))
    assert provider.generate("Complete source " * 100) == '{"verdict":"APPLY"}'
    assert "counted_text_tokens" not in provider.budget_diagnostics[-1]


@pytest.mark.parametrize("kwargs", [{"input_token_counter": 12}, {"keep_alive": -1},
                                    {"keep_alive": True}, {"keep_alive": "0"}])
def test_explicit_runtime_controls_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        OllamaProvider(client=FakeClient(), **kwargs)


def test_env_configuration_and_explicit_overrides(monkeypatch):
    for name, value in {"OLLAMA_MODEL": "qwen3:14b", "OLLAMA_BASE_URL": "http://127.0.0.1:11435", "OLLAMA_TIMEOUT_SECONDS": "900", "OLLAMA_NUM_CTX": "65536", "OLLAMA_NUM_PREDICT": "8192"}.items():
        monkeypatch.setenv(name, value)
    provider = OllamaProvider(client=FakeClient())
    assert (provider.model, provider.host, provider.timeout, provider.num_ctx, provider.num_predict) == (
        "qwen3:14b", "http://127.0.0.1:11435", 900.0, 65536, 8192,
    )
    override = OllamaProvider("qwen3:8b", "http://[::1]:11434", client=FakeClient(), timeout=120, num_ctx=16384, num_predict=2048)
    assert (override.model, override.host, override.timeout, override.num_ctx, override.num_predict) == (
        "qwen3:8b", "http://[::1]:11434", 120.0, 16384, 2048,
    )


def test_sdk_configuration_blocks_redirects_and_inherited_network_proxies(monkeypatch):
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        return FakeClient()
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=build))
    provider = OllamaProvider()
    assert provider.host == "http://127.0.0.1:11434"
    assert calls == [{"host": "http://127.0.0.1:11434", "timeout": 600.0, "follow_redirects": False, "trust_env": False,
                      "headers": {"Authorization": "Bearer local-only"}}]


@pytest.mark.parametrize("model", ["qwen3:cloud", "qwen3:8b-cloud", "MODEL:CLOUD", "", "  ", "qwen3:8b extra"])
def test_cloud_or_invalid_model_names_are_rejected(model):
    with pytest.raises(ValueError):
        OllamaProvider(model=model, client=FakeClient())


@pytest.mark.parametrize("host", [
    "https://ollama.com", "https://example.com", "http://192.168.1.2:11434", "http://0.0.0.0:11434",
    "http://localhost.example.com:11434", "http://localhost:11434/proxy", "http://user:secret@localhost:11434",
    "http://localhost:11434?destination=cloud", "ftp://localhost:11434", "http://localhost:0", "http://[fe80::1]:11434",
])
def test_nonlocal_or_credentialed_hosts_are_rejected(host):
    with pytest.raises(ValueError):
        OllamaProvider(host=host, client=FakeClient())


@pytest.mark.parametrize("setting", [
    {"timeout": 0}, {"timeout": float("nan")}, {"timeout": 7201}, {"timeout": True},
    {"num_ctx": 0}, {"num_ctx": 131073}, {"num_ctx": True},
    {"num_predict": 0}, {"num_predict": 32769}, {"num_predict": True},
    {"num_ctx": 4096, "num_predict": 4096},
])
def test_invalid_generation_limits_are_rejected(setting):
    with pytest.raises(ValueError):
        OllamaProvider(client=FakeClient(), **setting)


@pytest.mark.parametrize("name", ["OLLAMA_TIMEOUT_SECONDS", "OLLAMA_NUM_CTX", "OLLAMA_NUM_PREDICT"])
def test_malformed_env_generation_limits_do_not_leak_values(monkeypatch, name):
    monkeypatch.setenv(name, "private-bad-value")
    with pytest.raises(ValueError) as error:
        OllamaProvider(client=FakeClient())
    assert name in str(error.value)
    assert "private-bad-value" not in str(error.value)


def test_context_guard_rejects_before_request_without_removing_input():
    client = FakeClient()
    provider = OllamaProvider(client=client, num_ctx=2048, num_predict=1024)
    with pytest.raises(OllamaProviderError, match="no input was truncated"):
        provider.generate("A" * 513)
    assert not client.calls
    assert provider.usage["requests"] == 0


def test_context_guard_counts_utf8_bytes_and_system_schema():
    provider = OllamaProvider(client=FakeClient(), num_ctx=2048, num_predict=1024)
    with pytest.raises(OllamaProviderError, match="UTF-8 bytes"):
        provider.generate("界" * 171)
    with pytest.raises(OllamaProviderError):
        provider.generate_structured("A" * 400, Verdict, "Factual instructions.")
    assert provider.usage["requests"] == 0


@pytest.mark.parametrize("changes,message", [
    ({"done": False}, "incomplete response"), ({"done": None}, "incomplete response"),
    ({"done_reason": "length"}, "generation limit"), ({"done_reason": "load"}, "normal completion"),
    ({"done_reason": None}, "normal completion"), ({"message": {"content": ""}}, "no final answer"),
    ({"message": {"content": None}}, "no final answer"),
    ({"message": {"content": "result", "tool_calls": [{"name": "upload"}]}}, "tool request"),
    ({"error": "private server response"}, "returned an error"),
])
def test_partial_or_invalid_completion_never_produces_a_result(changes, message):
    provider = OllamaProvider(client=FakeClient(response(**changes)))
    with pytest.raises(OllamaProviderError, match=message) as error:
        provider.generate_structured("Prompt", Verdict)
    assert isinstance(error.value, ModelProviderError)
    assert "private server response" not in str(error.value)
    assert provider.usage["total_tokens"] == 120


def test_reported_prompt_count_near_context_limit_is_rejected():
    provider = OllamaProvider(client=FakeClient(response(prompt_eval_count=14000)), num_ctx=16384)
    with pytest.raises(OllamaProviderError, match="context limit"):
        provider.generate("small visible prompt")


@pytest.mark.parametrize("content", [
    '{"verdict":', '```json\n{"verdict":"APPLY"}\n```',
    '{"unexpected":"Private applicant data"}', '{"verdict":"APPLY"} trailing commentary',
])
def test_complete_json_and_requested_schema_are_required(content):
    provider = OllamaProvider(client=FakeClient(response(message={"content": content})))
    with pytest.raises(OllamaProviderError, match="not valid JSON") as error:
        provider.generate_structured("Prompt", Verdict)
    assert "Private applicant data" not in str(error.value)
    assert error.value.__suppress_context__ is True


def test_transport_error_is_sanitized_and_not_retried():
    failure = RuntimeError("Private CV and credentials in server diagnostics")
    failure.status_code = 404
    client = FakeClient(failure)
    provider = OllamaProvider(client=client)
    with pytest.raises(OllamaProviderError, match="ollama list") as error:
        provider.generate("Prompt")
    assert "Private CV" not in str(error.value)
    assert error.value.__suppress_context__ is True
    assert len(client.calls) == 1
    assert provider.usage["requests"] == 1


def test_timeout_error_is_actionable_without_exposing_prompt():
    class ReadTimeout(Exception):
        pass
    provider = OllamaProvider(client=FakeClient(ReadTimeout("Private CV")))
    with pytest.raises(OllamaProviderError, match="OLLAMA_TIMEOUT_SECONDS") as error:
        provider.generate("Prompt")
    assert "Private CV" not in str(error.value)


def test_attribute_style_response_is_supported():
    item = SimpleNamespace(done=True, done_reason="stop", message=SimpleNamespace(content="Ready", tool_calls=None), prompt_eval_count=5, eval_count=2)
    provider = OllamaProvider(client=FakeClient(item))
    assert provider.generate("Prompt") == "Ready"
    assert provider.usage["total_tokens"] == 7


def test_schema_validation_happens_before_any_local_call():
    provider = OllamaProvider(client=FakeClient())
    with pytest.raises(TypeError, match="BaseModel"):
        provider.generate_structured("Prompt", dict)
    assert provider.usage["requests"] == 0


def test_missing_sdk_has_local_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", None)
    with pytest.raises(ImportError, match="pip install ollama"):
        OllamaProvider()


def test_real_sdk_mock_transport_parses_response_without_network():
    ollama = pytest.importorskip("ollama")
    httpx = pytest.importorskip("httpx")
    observed = []
    def handle(request):
        assert request.url.host == "127.0.0.1"
        assert request.url.path == "/api/chat"
        observed.append(json.loads(request.content))
        return httpx.Response(200, json=response(model="qwen3:8b", created_at="2026-09-13T12:00:00Z"))
    client = ollama.Client(host="http://127.0.0.1:11434", transport=httpx.MockTransport(handle), trust_env=False)
    try:
        provider = OllamaProvider(client=client)
        assert provider.generate_structured("Check this result", Verdict) == Verdict(verdict="APPLY")
        assert observed[0]["think"] is False
        assert observed[0]["format"] == Verdict.model_json_schema()
        assert provider.usage == {"requests": 1, "input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    finally:
        client._client.close()
