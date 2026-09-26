"""Strict Gemma metadata and score-merging tests; no network or model loading."""

from copy import deepcopy
import itertools

import pytest

from app.providers import gemma_token_budget as budget


def payload():
    # Preserve real control-token positions; shrink only the normal vocabulary
    # and merge-table dimensions for bounded, fast, completely in-memory tests.
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
    tokens += [f"<unused{i}>" for i in range(101)]
    tokens += ["<start_of_turn>", "<end_of_turn>"]
    tokens += [f"<0x{i:02X}>" for i in range(256)]
    tokens += ["a", "b", "c", "ab", "bc", "abc", "▁", "▁a", "▁▁", "\t", "\n"]
    tokens += ["<image_soft_token>"]
    kinds = [1] * len(tokens)
    for index in (0, 1, 2, 105, 106):
        kinds[index] = 3
    kinds[3] = 2
    kinds[-1] = 4
    for index in range(107, 363):
        kinds[index] = 6
    scores = [-100.0] * len(tokens)
    for token, score in {"ab": -2, "bc": -1, "abc": -3, "▁a": -4, "▁▁": 0}.items():
        scores[tokens.index(token)] = score
    return {
        "details": {"family": "gemma3", "format": "gguf"},
        "template": budget._STANDARD_GEMMA3_TEMPLATE,
        "model_info": {
            "general.architecture": "gemma3",
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.pre": "default",
            "tokenizer.ggml.add_bos_token": True,
            "tokenizer.ggml.add_eos_token": False,
            "tokenizer.ggml.add_padding_token": False,
            "tokenizer.ggml.add_unknown_token": False,
            "tokenizer.ggml.bos_token_id": 2,
            "tokenizer.ggml.eos_token_id": 1,
            "tokenizer.ggml.padding_token_id": 0,
            "tokenizer.ggml.unknown_token_id": 3,
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": kinds,
            "tokenizer.ggml.scores": scores,
            "tokenizer.ggml.merges": ["a b", "b c", "ab c", "a bc", "▁ a", "▁ ▁"],
        },
    }


@pytest.fixture(autouse=True)
def small_dimensions(monkeypatch):
    source = payload()["model_info"]
    monkeypatch.setattr(budget, "_EXPECTED_VOCAB_SIZE", len(source["tokenizer.ggml.tokens"]))
    monkeypatch.setattr(budget, "_EXPECTED_MERGE_COUNT", len(source["tokenizer.ggml.merges"]))


def test_counts_text_only_and_preserves_source_and_messages():
    source = payload()
    before = deepcopy(source)
    count = budget.build_gemma_token_counter(source)
    messages = [{"role": "user", "content": "abc ab\t\n界"}]
    old_messages = deepcopy(messages)
    assert count(messages) == 8
    assert count([{"role": "user", "content": "abc"}]) == 1
    assert source == before
    assert messages == old_messages


def test_full_byte_fallback_preserves_unseen_unicode():
    count = budget.build_gemma_token_counter(payload())
    assert count([{"role": "user", "content": "界"}]) == 3
    assert count([{"role": "user", "content": "🧪"}]) == 4
    assert count([{"role": "user", "content": "é"}]) == 2


def test_spaces_are_escaped_without_whitespace_collapse_or_tab_rewrite():
    count = budget.build_gemma_token_counter(payload())
    assert count([{"role": "user", "content": "  a\t\n"}]) == 4
    assert count([{"role": "user", "content": "a\t\ta"}]) == 4
    assert count([{"role": "user", "content": "a\n\na"}]) == 4
    assert count([{"role": "user", "content": "a\u00a0a"}]) == 4


@pytest.mark.parametrize("field,value", [
    ("general.architecture", "gemma4"), ("tokenizer.ggml.model", "gpt2"),
    ("tokenizer.ggml.pre", "gemma3"), ("tokenizer.ggml.add_bos_token", False),
    ("tokenizer.ggml.add_bos_token", 1), ("tokenizer.ggml.add_eos_token", True),
    ("tokenizer.ggml.add_padding_token", None), ("tokenizer.ggml.add_unknown_token", True),
    ("tokenizer.ggml.bos_token_id", True), ("tokenizer.ggml.eos_token_id", 2),
    ("tokenizer.ggml.padding_token_id", -1), ("tokenizer.ggml.unknown_token_id", 0),
    ("tokenizer.ggml.tokens", None), ("tokenizer.ggml.tokens", []),
    ("tokenizer.ggml.scores", None), ("tokenizer.ggml.scores", [0]),
    ("tokenizer.ggml.token_type", None), ("tokenizer.ggml.token_type", [1]),
    ("tokenizer.ggml.merges", None), ("tokenizer.ggml.merges", []),
    ("tokenizer.ggml.add_space_prefix", False),
    ("tokenizer.ggml.remove_extra_whitespaces", True),
    ("tokenizer.ggml.escape_whitespaces", False),
    ("tokenizer.ggml.treat_whitespace_as_suffix", True),
    ("tokenizer.ggml.normalizer", "private replacement"),
    ("tokenizer.ggml.precompiled_charsmap", [1, 2]),
    ("tokenizer.custom", "private custom configuration"),
])
def test_rejects_unsupported_or_incomplete_metadata(field, value):
    source = payload()
    source["model_info"][field] = value
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_gemma_token_counter(source)
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("field,value", [
    ("details", None), ("details", {"family": "gemma4", "format": "gguf"}),
    ("details", {"family": "gemma3", "format": "private"}),
    ("model_info", None), ("system", "private hidden instructions"),
    ("messages", [{"role": "user", "content": "private history"}]),
    ("template", None), ("template", "{{.Messages}}{{.Messages}} private"),
    ("template", budget._STANDARD_GEMMA3_TEMPLATE + " private suffix"),
])
def test_rejects_hidden_content_or_custom_template(field, value):
    source = payload()
    source[field] = value
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        budget.build_gemma_token_counter(source)
    assert "private" not in str(exc.value)


def test_rejects_nonmapping_payload():
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_gemma_token_counter(None)


def test_only_outer_whitespace_and_crlf_template_variation_allowed():
    source = payload()
    source["template"] = "\n" + source["template"].replace("\n", "\r\n") + "  "
    assert budget.build_gemma_token_counter(source)([{"role": "user", "content": "a"}]) == 1


@pytest.mark.parametrize("field,value", [
    ("add_space_prefix", True), ("remove_extra_whitespaces", False),
    ("escape_whitespaces", True), ("treat_whitespace_as_suffix", False),
])
def test_explicit_standard_normalizer_defaults_allowed(field, value):
    source = payload()
    source["model_info"][f"tokenizer.ggml.{field}"] = value
    assert budget.build_gemma_token_counter(source)([{"role": "user", "content": "a"}]) == 1


@pytest.mark.parametrize("score", [True, None, "0", float("nan"), float("inf"),
                                    float("-inf"), 1, -(10 ** 1000), 10 ** 1000])
def test_rejects_invalid_scores(score):
    source = payload()
    source["model_info"]["tokenizer.ggml.scores"][120] = score
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_gemma_token_counter(source)


@pytest.mark.parametrize("token", [None, True, "", "x" * 4097, "\ud800", "<bos>"])
def test_rejects_invalid_or_duplicate_vocabulary_tokens(token):
    source = payload()
    source["model_info"]["tokenizer.ggml.tokens"][10] = token
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_gemma_token_counter(source)


@pytest.mark.parametrize("kind", [True, None, "1", 0, 2, 3, 4, 5, 6, 7])
def test_rejects_invalid_types_and_unexpected_special_or_byte_tokens(kind):
    source = payload()
    source["model_info"]["tokenizer.ggml.token_type"][10] = kind
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_gemma_token_counter(source)


def test_missing_byte_is_rejected_even_when_dimensions_match():
    source = payload()
    source["model_info"]["tokenizer.ggml.tokens"][120] = "missing byte replacement"
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_gemma_token_counter(source)


def test_missing_space_piece_is_rejected():
    source = payload()
    tokens = source["model_info"]["tokenizer.ggml.tokens"]
    tokens[tokens.index("▁")] = "replacement"
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_gemma_token_counter(source)


@pytest.mark.parametrize("merge", [None, True, "", "a missing", "a b c", "b b", "a bc", "x" * 8194])
def test_rejects_invalid_or_duplicate_merges(merge):
    source = payload()
    source["model_info"]["tokenizer.ggml.merges"][0] = merge
    with pytest.raises(budget.LocalTokenBudgetError):
        budget.build_gemma_token_counter(source)


@pytest.mark.parametrize("messages", [
    None, [], ["private"], [{"role": "system", "content": "a"}],
    [{"role": "assistant", "content": "a"}], [{"role": "tool", "content": "a"}],
    [{"role": "user", "content": "a"}] * 2,
    [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "user", "content": "a", "images": ["private"]}],
    [{"role": "user", "content": "a", "tools": []}],
    [{"role": "user", "content": None}], [{"role": "user", "content": ""}],
    [{"role": "user", "content": "\t\n "}],
    [{"role": "user", "content": "a" * 1_000_001}],
])
def test_unsupported_message_shapes_fail_closed(messages):
    counter = budget.build_gemma_token_counter(payload())
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        counter(messages)
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("reserved", [
    "<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>",
    "<end_of_turn>", "<image_soft_token>",
])
def test_reserved_model_literals_rejected_instead_of_guessed(reserved):
    counter = budget.build_gemma_token_counter(payload())
    with pytest.raises(budget.LocalTokenBudgetError):
        counter([{"role": "user", "content": f"private {reserved} data"}])


@pytest.mark.parametrize("value", [None, True, 0, -1, "1", 100])
def test_invalid_counter_results_fail_closed(monkeypatch, value):
    counter = budget.build_gemma_token_counter(payload())
    monkeypatch.setattr(budget, "_count_spm_text", lambda *args: value)
    with pytest.raises(budget.LocalTokenBudgetError):
        counter([{"role": "user", "content": "abc"}])


def test_counter_exceptions_do_not_leak_content(monkeypatch):
    counter = budget.build_gemma_token_counter(payload())
    def fail(*args):
        raise RuntimeError("private resume text")
    monkeypatch.setattr(budget, "_count_spm_text", fail)
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        counter([{"role": "user", "content": "private resume text"}])
    assert "private" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_invalid_unicode_is_sanitized():
    counter = budget.build_gemma_token_counter(payload())
    with pytest.raises(budget.LocalTokenBudgetError) as exc:
        counter([{"role": "user", "content": "private \ud800"}])
    assert "private" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_counter_isolated_from_metadata_mutation_after_construction():
    source = payload()
    counter = budget.build_gemma_token_counter(source)
    source["model_info"]["tokenizer.ggml.tokens"].clear()
    source["model_info"]["tokenizer.ggml.scores"].clear()
    assert counter([{"role": "user", "content": "abc"}]) == 1


def test_score_order_and_leftmost_ties_match_naive_merge_reference():
    # This exercises invalidated queue entries, repeated symbols, Unicode byte
    # fallback, and ties. Unigram/Viterbi segmentation is intentionally not used.
    vocab = {token: i for i, token in enumerate(["a", "b", "c", "ab", "bc", "abc", "aa", "aab", "▁"])}
    scores = (-100.0, -100.0, -100.0, -2.0, -2.0, -3.0, -1.0, -4.0, -100.0)
    def naive(text):
        symbols = list(text.replace(" ", "▁"))
        while True:
            options = [(-scores[vocab[symbols[i] + symbols[i + 1]]], i)
                       for i in range(len(symbols) - 1) if symbols[i] + symbols[i + 1] in vocab]
            if not options:
                break
            _, left = min(options)
            symbols[left:left + 2] = [symbols[left] + symbols[left + 1]]
        return sum(1 if token in vocab else len(token.encode()) for token in symbols)
    for size in range(1, 6):
        for chars in itertools.product("abc 界", repeat=size):
            content = "".join(chars)
            assert budget._count_spm_text(content, vocab, scores) == naive(content)
