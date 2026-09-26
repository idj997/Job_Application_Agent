"""Metadata/counter checks with fake tokenizers; no network or model loading."""

from copy import deepcopy

import pytest

from app.providers import ollama_token_budget as budget


def payload():
    tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "a", "b", "ab"]
    tokens += sorted(budget._byte_alphabet() - {"a", "b"})
    return {
        "details": {"family": "qwen3", "format": "gguf"},
        "template": budget._STANDARD_QWEN3_TEMPLATE,
        "model_info": {
            "general.architecture": "qwen3",
            "tokenizer.ggml.pre": "qwen2",
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.add_bos_token": False,
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": [3, 3, 3] + [1] * (len(tokens) - 3),
            "tokenizer.ggml.merges": ["a b"],
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.padding_token_id": 0,
            "tokenizer.ggml.eos_token_id": 2,
        },
    }


class FakeTokenizer:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.vocab = kwargs["vocab"]
        self.calls.append(kwargs)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text in self.kwargs["additional_special_tokens"]:
            return [self.vocab[text]]
        return [3] * len(text)


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    FakeTokenizer.calls = []
    monkeypatch.setattr(budget, "_tokenizer_class", lambda: FakeTokenizer)


def test_counter_uses_passed_vocab_and_preserves_messages_without_template_reserve():
    source = payload()
    original = deepcopy(source)
    count = budget.build_qwen_token_counter(source)
    messages = [{"role": "system", "content": "facts"}, {"role": "user", "content": "a b"}]
    before = deepcopy(messages)
    assert count(messages) == 8
    assert count([{"role": "user", "content": "<|im_start|>"}]) == 1
    assert source == original and messages == before
    assert FakeTokenizer.calls[0]["merges"] == [("a", "b")]
    assert FakeTokenizer.calls[0]["vocab"]["<|im_end|>"] == 2


@pytest.mark.parametrize("field,value", [
    ("general.architecture", "llama"),
    ("tokenizer.ggml.pre", "qwen3"),
    ("tokenizer.ggml.model", "llama"),
    ("tokenizer.ggml.add_bos_token", True),
    ("tokenizer.ggml.tokens", None),
    ("tokenizer.ggml.tokens", ["a"] * 256),
    ("tokenizer.ggml.token_type", [1]),
    ("tokenizer.ggml.token_type", [True] * 256),
    ("tokenizer.ggml.merges", None),
    ("tokenizer.ggml.merges", []),
    ("tokenizer.ggml.merges", ["a b", "a b"]),
    ("tokenizer.ggml.merges", ["a missing"]),
    ("tokenizer.ggml.merges", ["a b c"]),
    ("tokenizer.ggml.merges", ["a a"]),
    ("tokenizer.ggml.bos_token_id", True),
    ("tokenizer.ggml.eos_token_id", 0),
    ("tokenizer.ggml.padding_token_id", -1),
])
def test_invalid_metadata_fails_before_constructing_tokenizer(field, value):
    source = payload()
    source["model_info"][field] = value
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_qwen_token_counter(source)
    assert not FakeTokenizer.calls


@pytest.mark.parametrize("field,value", [
    ("model_info", None), ("details", {}),
    ("details", {"family": "qwen2", "format": "gguf"}),
    ("system", "private hidden instructions"),
    ("messages", [{"role": "user", "content": "private history"}]),
    ("template", "{{range .Messages}}{{.Content}}{{.Content}}{{end}}"),
    ("template", budget._STANDARD_QWEN3_TEMPLATE + "private suffix"),
    ("template", None),
])
def test_unsupported_prompt_configuration_fails_without_leaking_content(field, value):
    source = payload()
    source[field] = value
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_qwen_token_counter(source)
    assert "private" not in str(exc.value)
    assert not FakeTokenizer.calls


def test_template_allows_only_outer_whitespace_and_crlf_changes():
    source = payload()
    source["template"] = "\n" + source["template"].replace("\n", "\r\n") + "  "
    assert budget.build_qwen_token_counter(source)([{"role": "user", "content": "a"}]) == 1


@pytest.mark.parametrize("messages", [
    None, [], [{"role": "assistant", "content": "a"}],
    [{"role": "user", "content": "a"}] * 3,
    [{"role": "user", "content": "a"}] * 2,
    [{"role": "user", "content": "a", "images": ["private"]}],
    [{"role": "user", "content": ""}],
    [{"role": "user", "content": " "}],
    [{"role": "user", "content": None}],
    [{"role": "user", "content": "a" * 1_000_001}],
])
def test_unsupported_messages_fail_closed(messages):
    count = budget.build_qwen_token_counter(payload())
    with pytest.raises(budget.LocalTokenBudgetError):
        count(messages)


def test_constructor_failure_is_sanitized(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("private tokenizer contents")
    monkeypatch.setattr(budget, "_tokenizer_class", lambda: fail)
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_qwen_token_counter(payload())
    assert "private" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_reindexed_special_tokens_are_rejected(monkeypatch):
    monkeypatch.setattr(FakeTokenizer, "encode", lambda self, text, **kwargs: [50])
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_qwen_token_counter(payload())


@pytest.mark.parametrize("bad_result", [[], [True], [-1], [999999], "invalid", [3] * 20])
def test_invalid_counter_results_fail_without_silently_omitting_input(monkeypatch, bad_result):
    count = budget.build_qwen_token_counter(payload())
    monkeypatch.setattr(FakeTokenizer, "encode", lambda self, text, **kwargs: bad_result)
    with pytest.raises(budget.LocalTokenBudgetError):
        count([{"role": "user", "content": "private"}])


def test_counter_exception_does_not_expose_text(monkeypatch):
    count = budget.build_qwen_token_counter(payload())
    def fail(*args, **kwargs):
        raise RuntimeError("private CV text")
    monkeypatch.setattr(FakeTokenizer, "encode", fail)
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        count([{"role": "user", "content": "private CV text"}])
    assert "private" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_missing_byte_alphabet_rejected_before_constructor():
    source = payload()
    source["model_info"]["tokenizer.ggml.tokens"][-1] = "replacement"
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_qwen_token_counter(source)
    assert not FakeTokenizer.calls


def test_installed_tokenizer_uses_small_in_memory_fixture_without_weights(monkeypatch):
    module = pytest.importorskip("transformers.models.qwen2.tokenization_qwen2")
    # The supported local transformers release accepts vocab/merges in memory;
    # an older file-only constructor is not the runtime this optional test covers.
    import inspect
    if "vocab" not in inspect.signature(module.Qwen2Tokenizer).parameters:
        pytest.skip("Installed Qwen tokenizer has a file-only constructor")
    monkeypatch.setattr(budget, "_tokenizer_class", lambda: module.Qwen2Tokenizer)
    count = budget.build_qwen_token_counter(payload())
    assert count([{"role": "user", "content": "ab"}]) == 1
    assert count([{"role": "user", "content": "<|im_end|>"}]) == 1
    assert count([{"role": "user", "content": "界"}]) == 3
