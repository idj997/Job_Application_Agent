"""No model downloads or inference: role runtime isolation and identity gates."""
from pydantic import BaseModel
import pytest

from app.providers.local_critic_runtime import LocalRoleProvider, local_role_config
from app.providers.ollama_provider import OllamaProviderError


class Result(BaseModel):
    answer: str


class Client:
    def __init__(self, family="gemma3", model="gemma3:4b"):
        self.family, self.model = family, model
        self.digest = "original-digest"
        self.calls = []
        self.change_on_show = False
        self.change_on_chat = False

    def list(self):
        return {"models": [{"model": self.model, "digest": self.digest}]}

    def show(self, model, verbose):
        assert model == self.model and verbose is True
        if self.change_on_show:
            self.digest = "replacement-digest"
        return {"details": {"family": self.family}}

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.change_on_chat:
            self.digest = "replacement-digest"
        return {"done": True, "done_reason": "stop", "message": {"content": '{"answer":"ok"}'},
                "prompt_eval_count": 110, "eval_count": 10}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr("app.providers.ollama_provider._load_environment", lambda: None)
    for name in ("OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_NUM_CTX", "OLLAMA_NUM_PREDICT",
                 "OLLAMA_TIMEOUT_SECONDS", "CV_CRITIC_NUM_CTX", "CV_CRITIC_NUM_PREDICT",
                 "CV_CRITIC_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("app.providers.gemma_token_budget.build_gemma_token_counter", lambda _: lambda messages: 100)
    monkeypatch.setattr("app.providers.ollama_token_budget.build_qwen_token_counter", lambda _: lambda messages: 100)


def test_pure_config_and_lazy_provider():
    class NoCalls:
        def __getattr__(self, name):
            raise AssertionError("Unexpected client activity")
    provider = LocalRoleProvider("gemma3:4b", "critic", client=NoCalls())
    assert provider.usage["requests"] == 0
    assert provider.model_digest is None
    assert provider.budget_diagnostics == []
    assert provider.fingerprint == local_role_config("gemma3:4b", "critic")
    assert (provider.num_ctx, provider.num_predict) == (12288, 4096)
    assert local_role_config("qwen3:8b", "writer")["num_predict"] == 4096


def test_older_sdk_verbose_metadata_uses_existing_transport():
    class OlderClient(Client):
        def _request_raw(self, method, path, **kwargs):
            assert method == "POST" and path == "/api/show"
            assert kwargs == {"json": {"model": "gemma3:4b", "verbose": True}}
            from types import SimpleNamespace
            return SimpleNamespace(json=lambda: {"details": {"family": "gemma3"}})

        def show(self, model):
            raise AssertionError("Old SDK show cannot request full vocabulary")
    client = OlderClient()
    assert LocalRoleProvider("gemma3:4b", "critic", client=client).generate_structured("Source", Result).answer == "ok"


def test_raw_metadata_does_not_discard_hidden_system_prompt(monkeypatch):
    class RawClient(Client):
        def _request_raw(self, *args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(json=lambda: {"details": {"family": "gemma3"},
                                                  "system": "Hidden instructions", "messages": []})
    def reject_hidden(payload):
        assert payload["system"] == "Hidden instructions"
        assert payload["messages"] == []
        raise ValueError("Custom hidden prompt")
    monkeypatch.setattr("app.providers.gemma_token_budget.build_gemma_token_counter", reject_hidden)
    client = RawClient()
    with pytest.raises(OllamaProviderError, match="standard template"):
        LocalRoleProvider("gemma3:4b", "critic", client=client).generate("Source")
    assert not client.calls


@pytest.mark.parametrize("role,family,model", [
    ("critic", "gemma3", "gemma3:4b"),
    ("writer", "qwen3", "qwen3:8b"), ("critic", "qwen3", "qwen3:8b"),
    ("writer", "qwen3", "qwen3:14b"), ("critic", "qwen3", "qwen3:14b"),
    ("critic", "mistral3", "ministral-3:8b"), ("writer", "mistral3", "ministral-3:8b"),
    ("critic", "mistral3", "ministral-3:14b"), ("writer", "mistral3", "ministral-3:14b"),
])
def test_digest_bound_sequential_runtime(role, family, model, monkeypatch):
    if family == "mistral3":
        import sys
        from types import SimpleNamespace
        monkeypatch.setitem(sys.modules, "app.providers.ministral_token_budget",
                            SimpleNamespace(build_ministral_token_counter=lambda _: lambda messages: 100))
    client = Client(family, model)
    provider = LocalRoleProvider(model, role, client=client)
    original_fingerprint = dict(provider.fingerprint)
    assert provider.generate_structured("Complete sources", Result).answer == "ok"
    call = client.calls[0]
    assert call["model"] == model
    assert call["keep_alive"] == 0
    assert call["options"]["num_thread"] == 6
    assert call["options"]["num_batch"] == 128
    assert ("think" in call) == (family == "qwen3")
    if family == "qwen3":
        assert call["think"] is False
    assert len(call["messages"]) == (1 if family == "gemma3" else 2)
    assert provider.model_digest == "original-digest"
    assert provider.fingerprint == original_fingerprint
    assert provider.usage["requests"] == 1


@pytest.mark.parametrize("role", ["writer", "critic"])
@pytest.mark.parametrize("model", ["ministral-3:8b", "ministral-3:14b"])
def test_ministral_keeps_explicit_system_and_never_uses_template_default(monkeypatch, role, model):
    import sys
    from types import SimpleNamespace
    observed = []
    def build(payload):
        assert payload["details"]["family"] == "mistral3"
        def counter(messages):
            observed.append(messages)
            if [message["role"] for message in messages] != ["system", "user"]:
                raise ValueError("Explicit system required")
            return 100
        return counter
    monkeypatch.setitem(sys.modules, "app.providers.ministral_token_budget",
                        SimpleNamespace(build_ministral_token_counter=build))
    client = Client("mistral3", model)
    provider = LocalRoleProvider(model, role, client=client)
    assert provider.generate_structured("Full source data", Result, "Source facts only.").answer == "ok"
    call = client.calls[0]
    assert call["messages"] == observed[0]
    assert call["messages"][0]["content"].startswith("Source facts only.")
    assert call["messages"][1] == {"role": "user", "content": "Full source data"}
    assert "think" not in call
    with pytest.raises(OllamaProviderError, match="could not count"):
        provider.generate("Plain message without system instructions")
    assert len(client.calls) == 1


@pytest.mark.parametrize("role", ["writer", "critic"])
@pytest.mark.parametrize("family,model", [("qwen3", "qwen3:14b"), ("mistral3", "ministral-3:14b")])
def test_larger_tags_do_not_bypass_cached_tokenizer_and_template_validation(monkeypatch, role, family, model):
    import sys
    from types import SimpleNamespace

    inspected = []

    def reject_metadata(payload):
        inspected.append(payload)
        assert payload["details"]["family"] == family
        raise ValueError("Cached tokenizer or template was not verified")

    if family == "mistral3":
        monkeypatch.setitem(sys.modules, "app.providers.ministral_token_budget",
                            SimpleNamespace(build_ministral_token_counter=reject_metadata))
    else:
        monkeypatch.setattr("app.providers.ollama_token_budget.build_qwen_token_counter", reject_metadata)
    client = Client(family, model)
    provider = LocalRoleProvider(model, role, client=client)
    with pytest.raises(OllamaProviderError, match="standard template"):
        provider.generate_structured("Complete assigned sources", Result)
    assert len(inspected) == 1
    assert client.calls == [] and provider.usage["requests"] == 0
    assert provider.model_digest is None


@pytest.mark.parametrize("model,role,prefix", [
    ("ministral-3:8b", "writer", "OLLAMA"), ("ministral-3:14b", "writer", "OLLAMA"),
    ("qwen3:8b", "critic", "CV_CRITIC"), ("qwen3:14b", "critic", "CV_CRITIC"),
])
def test_reversed_models_select_limits_by_role_not_family(monkeypatch, model, role, prefix):
    for name, value in {
        "OLLAMA_NUM_CTX": "8192", "OLLAMA_NUM_PREDICT": "512", "OLLAMA_TIMEOUT_SECONDS": "120",
        "CV_CRITIC_NUM_CTX": "16384", "CV_CRITIC_NUM_PREDICT": "2048", "CV_CRITIC_TIMEOUT_SECONDS": "300",
    }.items():
        monkeypatch.setenv(name, value)
    config = local_role_config(model, role)
    assert (config["num_ctx"], config["num_predict"], config["timeout_seconds"]) == (
        (8192, 512, 120) if prefix == "OLLAMA" else (16384, 2048, 300)
    )
    assert config["model"] == model and config["role"] == role
    assert config["keep_alive"] == 0


@pytest.mark.parametrize("when", ["show", "before_chat", "chat"])
def test_changed_model_rejected(when):
    client = Client()
    provider = LocalRoleProvider("gemma3:4b", "critic", client=client)
    if when == "show":
        client.change_on_show = True
    elif when == "before_chat":
        provider._load()
        client.digest = "replacement-digest"
    else:
        client.change_on_chat = True
    with pytest.raises(OllamaProviderError):
        provider.generate_structured("Complete sources", Result)
    assert len(client.calls) == (1 if when == "chat" else 0)


@pytest.mark.parametrize("family,installed", [("unsupported", True), ("gemma3", False)])
def test_unsupported_or_uninstalled_model_never_infers(family, installed):
    client = Client(family, "gemma3:4b" if installed else "different:4b")
    with pytest.raises(OllamaProviderError):
        LocalRoleProvider("gemma3:4b", "critic", client=client).generate("source")
    assert not client.calls


@pytest.mark.parametrize("name,value", [("CV_CRITIC_NUM_CTX", "bad"), ("CV_CRITIC_NUM_CTX", "1024"),
                                       ("CV_CRITIC_TIMEOUT_SECONDS", "nan"), ("CV_CRITIC_NUM_PREDICT", "0"),
                                       ("OLLAMA_BASE_URL", "https://external.example")])
def test_invalid_configuration_fails_without_network(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        local_role_config("gemma3:4b", "critic")
