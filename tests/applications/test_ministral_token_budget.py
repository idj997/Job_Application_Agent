"""Strict cached Ministral/Tekken counters: offline, no model loading."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from app.providers import ministral_token_budget as budget


# Small in-memory fixture, not a substitute production template. Its digest and
# normal-vocabulary dimensions are patched only inside these offline tests.
_FIXTURE_TEMPLATE = '''{{- range .Messages }}
{{- if eq .Role "system" }}[SYSTEM_PROMPT]{{ .Content }}[/SYSTEM_PROMPT]
{{- else if eq .Role "user" }}[INST]{{ .Content }}[/INST]{{- end }}{{- end }}'''
_FIXTURE_14B_TEMPLATE = '{{- $identity := "Ministral-3-14B-Instruct-2512" }}\n' + _FIXTURE_TEMPLATE
_PUBLISHED_TEMPLATE_HASHES = (
    budget._STANDARD_MINISTRAL_TEMPLATE_SHA256,
    budget._STANDARD_MINISTRAL_14B_TEMPLATE_SHA256,
)


def payload():
    tokens = list(budget._special_tokens()) + sorted(budget._byte_alphabet())
    tokens += ["ab", "bc", "abc", "Ġa", "ĠĠ", "2024"]
    return {
        "details": {"family": "mistral3", "format": "gguf"},
        "template": _FIXTURE_TEMPLATE,
        "model_info": {
            "general.architecture": "mistral3",
            "mistral3.vocab_size": len(tokens),
            "mistral3.rope.scaling_beta": 0.1,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "default",
            "tokenizer.ggml.add_bos_token": True,
            "tokenizer.ggml.add_eos_token": False,
            "tokenizer.ggml.add_padding_token": False,
            "tokenizer.ggml.add_unknown_token": False,
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.padding_token_id": 11,
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": [3] * 1000 + [1] * (len(tokens) - 1000),
            "tokenizer.ggml.scores": list(range(len(tokens))),
            "tokenizer.ggml.merges": ["a b", "b c", "Ġ a", "Ġ Ġ"],
        },
    }


def messages(content="abc"):
    return [{"role": "system", "content": "a"}, {"role": "user", "content": content}]


@pytest.fixture(autouse=True)
def small_metadata(monkeypatch):
    source = payload()["model_info"]
    monkeypatch.setattr(budget, "_EXPECTED_VOCAB_SIZE", len(source["tokenizer.ggml.tokens"]))
    monkeypatch.setattr(budget, "_EXPECTED_MERGE_COUNT", len(source["tokenizer.ggml.merges"]))
    monkeypatch.setattr(budget, "_STANDARD_MINISTRAL_TEMPLATE_SHA256", hashlib.sha256(_FIXTURE_TEMPLATE.encode()).hexdigest())


def test_uses_cached_vocabulary_and_tekken_whole_piece_lookup_without_mutation():
    source, request = payload(), messages()
    source_before, request_before = deepcopy(source), deepcopy(request)
    count = budget.build_ministral_token_counter(source)
    # No merge pair produces abc in the fixture. Tekken's direct vocabulary
    # lookup (ignore_merges) must still encode the complete regex piece once.
    assert count(request) == 2
    assert count(messages("ab")) == 2
    assert source == source_before
    assert request == request_before


def test_regex_splits_individual_digits_even_when_full_number_in_vocabulary():
    count = budget.build_ministral_token_counter(payload())
    assert count(messages("2024")) == 5


@pytest.mark.parametrize("content,expected", [
    ("界", 4), ("🧪", 5), ("é", 3), ("a\x00b", 4),
    ("a\t\ta", 5), ("a\n\na", 5), ("a\u00a0a", 5),
    ("  a\t\n", 5),
])
def test_all_bytes_unicode_and_whitespace_are_retained(content, expected):
    assert budget.build_ministral_token_counter(payload())(messages(content)) == expected


@pytest.mark.parametrize("field,value", [
    ("general.architecture", "llama"), ("mistral3.vocab_size", True),
    ("mistral3.vocab_size", 131072), ("tokenizer.ggml.model", "llama"),
    ("tokenizer.ggml.pre", "qwen2"), ("tokenizer.ggml.pre", []),
    ("tokenizer.ggml.add_bos_token", False), ("tokenizer.ggml.add_bos_token", 1),
    ("tokenizer.ggml.add_eos_token", True), ("tokenizer.ggml.add_padding_token", None),
    ("tokenizer.ggml.add_unknown_token", True),
    ("tokenizer.ggml.bos_token_id", True), ("tokenizer.ggml.eos_token_id", 1),
    ("tokenizer.ggml.padding_token_id", 0), ("tokenizer.ggml.unknown_token_id", 1),
    ("tokenizer.ggml.tokens", None), ("tokenizer.ggml.tokens", []),
    ("tokenizer.ggml.token_type", None), ("tokenizer.ggml.token_type", [1]),
    ("tokenizer.ggml.scores", None), ("tokenizer.ggml.scores", []),
    ("tokenizer.ggml.merges", None), ("tokenizer.ggml.merges", []),
    ("tokenizer.ggml.add_space_prefix", True),
    ("tokenizer.ggml.remove_extra_whitespaces", True),
    ("tokenizer.ggml.escape_whitespaces", True),
    ("tokenizer.ggml.treat_whitespace_as_suffix", True),
    ("tokenizer.ggml.normalizer", "private replacement"),
    ("tokenizer.ggml.precompiled_charsmap", [1, 2]),
    ("tokenizer.custom", "private custom tokenizer"),
])
def test_rejects_incomplete_custom_or_unsupported_metadata(field, value):
    source = payload()
    source["model_info"][field] = value
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_ministral_token_counter(source)
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("field,value", [
    ("details", None), ("details", {"family": "llama", "format": "gguf"}),
    ("details", {"family": "mistral3", "format": "private"}),
    ("model_info", None), ("system", "private hidden prompt"),
    ("messages", [{"role": "user", "content": "private history"}]),
    ("template", None), ("template", "{{ .Messages }}{{ .Messages }}"),
    ("template", _FIXTURE_TEMPLATE + " private hidden suffix"),
    ("template", "\ud800"), ("template", "x" * 8193),
])
def test_rejects_hidden_content_and_custom_templates(field, value):
    source = payload()
    source[field] = value
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_ministral_token_counter(source)
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("source", [None, [], 1, "private"])
def test_rejects_nonmapping_payload(source):
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)


def test_only_outer_whitespace_and_crlf_template_variations_allowed():
    source = payload()
    source["template"] = "\n" + _FIXTURE_TEMPLATE.replace("\n", "\r\n") + "  "
    assert budget.build_ministral_token_counter(source)(messages()) == 2


def test_production_template_hashes_are_exactly_the_two_inspected_releases():
    assert _PUBLISHED_TEMPLATE_HASHES == (
        "ab2bb6a5d55d3857ed25790b2ea7f27e9a21e1478ffd6a37a7ca79dce38a13e1",
        "4789ab48bdc47ce9a0ee0bd3801eff135a329b7d53398a348e901315309f0a63",
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_explicitly_allowlisted_14b_template_preserves_counter_behavior(monkeypatch, normalize):
    monkeypatch.setattr(budget, "_STANDARD_MINISTRAL_14B_TEMPLATE_SHA256",
                        hashlib.sha256(_FIXTURE_14B_TEMPLATE.encode()).hexdigest())
    source = payload()
    source["template"] = _FIXTURE_14B_TEMPLATE
    if normalize:
        source["template"] = "\n" + source["template"].replace("\n", "\r\n") + "  "
    assert budget.build_ministral_token_counter(source)(messages()) == 2
    # The previously inspected 8B branch remains accepted independently.
    assert budget.build_ministral_token_counter(payload())(messages()) == 2


@pytest.mark.parametrize("mutation", [
    lambda source: source.update(template=_FIXTURE_14B_TEMPLATE.replace("14B", "3B")),
    lambda source: source.update(template=_FIXTURE_14B_TEMPLATE + " hidden instructions"),
    lambda source: source.update(template=_FIXTURE_14B_TEMPLATE.replace("{{ .Content }}", "{{ .Content }}{{ .Content }}")),
    lambda source: source.update(system="hidden instructions"),
    lambda source: source.update(messages=[{"role": "system", "content": "hidden instructions"}]),
    lambda source: source["model_info"].update({"tokenizer.ggml.merges": []}),
    lambda source: source["model_info"].update({"tokenizer.ggml.tokens": []}),
])
def test_larger_template_does_not_relax_custom_template_or_metadata_checks(monkeypatch, mutation):
    monkeypatch.setattr(budget, "_STANDARD_MINISTRAL_14B_TEMPLATE_SHA256",
                        hashlib.sha256(_FIXTURE_14B_TEMPLATE.encode()).hexdigest())
    source = payload()
    source["template"] = _FIXTURE_14B_TEMPLATE
    mutation(source)
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)


def test_default_tokenizer_requires_ollama_compatibility_detection_marker():
    source = payload()
    del source["model_info"]["mistral3.rope.scaling_beta"]
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)
    source["model_info"]["tokenizer.ggml.pre"] = "tekken"
    assert budget.build_ministral_token_counter(source)(messages()) == 2


@pytest.mark.parametrize("field", ["add_space_prefix", "remove_extra_whitespaces", "escape_whitespaces", "treat_whitespace_as_suffix"])
def test_explicit_standard_normalizer_settings_allowed(field):
    source = payload()
    source["model_info"]["tokenizer.ggml." + field] = False
    assert budget.build_ministral_token_counter(source)(messages()) == 2


@pytest.mark.parametrize("token", [None, True, "", "x" * 4097, "\ud800", "<s>"])
def test_rejects_invalid_or_duplicate_tokens(token):
    source = payload()
    source["model_info"]["tokenizer.ggml.tokens"][-1] = token
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)


@pytest.mark.parametrize("score", [None, True, "1", -1, 0.5, float("nan"), float("inf"), float("-inf"), 10 ** 1000])
def test_rejects_invalid_or_altered_export_scores(score):
    source = payload()
    source["model_info"]["tokenizer.ggml.scores"][1100] = score
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)


@pytest.mark.parametrize("kind", [None, True, "1", 0, 2, 3, 4, 5, 6, 7])
def test_rejects_invalid_normal_token_types(kind):
    source = payload()
    source["model_info"]["tokenizer.ggml.token_type"][1100] = kind
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)


def test_missing_byte_rejected_even_with_correct_dimensions():
    source = payload()
    source["model_info"]["tokenizer.ggml.tokens"][1000] = "missing-byte"
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)


@pytest.mark.parametrize("index", [0, 1, 17, 34, 999])
def test_changed_special_tokens_rejected(index):
    source = payload()
    source["model_info"]["tokenizer.ggml.tokens"][index] = "private unexpected control"
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)


@pytest.mark.parametrize("merge", [None, True, "", "a missing", "a b c", "b b", "b c", "<s> a", "x" * 8194])
def test_invalid_incomplete_duplicate_or_special_merges_rejected(merge):
    source = payload()
    source["model_info"]["tokenizer.ggml.merges"][0] = merge
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(source)


@pytest.mark.parametrize("message_list", [
    None, [], ["private"], [{"role": "user", "content": "abc"}],
    [{"role": "system", "content": "a"}],
    [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}],
    [{"role": "system", "content": "a"}, {"role": "user", "content": "b", "images": []}],
    messages(None), messages(""), messages(" \t\n"), messages("a" * 1_000_001),
    messages() + [{"role": "assistant", "content": "c"}],
])
def test_unsupported_message_shapes_fail_closed(message_list):
    counter = budget.build_ministral_token_counter(payload())
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        counter(message_list)
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("token", ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[THINK]", "[IMG]", "<SPECIAL_998>"])
def test_control_literals_rejected_in_both_messages(token):
    counter = budget.build_ministral_token_counter(payload())
    for index in (0, 1):
        request = messages()
        request[index]["content"] = "private " + token
        with pytest.raises(budget.LocalTokenBudgetError):
            counter(request)


def test_invalid_unicode_sanitized():
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_ministral_token_counter(payload())(messages("private \ud800"))
    assert "private" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_metadata_mutation_after_construction_cannot_alter_counter():
    source = payload()
    counter = budget.build_ministral_token_counter(source)
    for key in ("tokens", "scores", "token_type", "merges"):
        source["model_info"]["tokenizer.ggml." + key].clear()
    assert counter(messages()) == 2


@pytest.mark.parametrize("failure", [ImportError, RuntimeError, ValueError])
def test_constructor_failures_sanitized_without_fallback(monkeypatch, failure):
    def fail(*args):
        raise failure("private constructor data")
    monkeypatch.setattr(budget, "_build_tokenizer", fail)
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_ministral_token_counter(payload())
    assert "private" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_reindexed_tokenizer_rejected(monkeypatch):
    monkeypatch.setattr(budget, "_build_tokenizer", lambda *args: SimpleNamespace(get_vocab=lambda: {}))
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(payload())


@pytest.mark.parametrize("ids", [None, [], [True], [-1], [0], [999], [999999], [1100] * 10])
def test_invalid_counter_outputs_rejected(monkeypatch, ids):
    def build(vocab, merges):
        return SimpleNamespace(get_vocab=lambda: vocab, encode=lambda *a, **kw: SimpleNamespace(ids=ids), decode=lambda *a, **kw: "a")
    monkeypatch.setattr(budget, "_build_tokenizer", build)
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_ministral_token_counter(payload())(messages())


def test_round_trip_failure_cannot_silently_omit_input(monkeypatch):
    def build(vocab, merges):
        return SimpleNamespace(get_vocab=lambda: vocab,
            encode=lambda *a, **kw: SimpleNamespace(ids=[vocab["a"]]),
            decode=lambda *a, **kw: "private altered input")
    monkeypatch.setattr(budget, "_build_tokenizer", build)
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_ministral_token_counter(payload())(messages())
    assert "private" not in str(exc.value)
