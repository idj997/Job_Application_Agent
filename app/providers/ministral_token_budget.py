"""Bounded text-token estimates from installed Ministral 3 8B/14B vocabularies.

This module never reads files, makes requests, loads a model or downloads a
tokenizer. Its caller supplies complete local ``show(verbose=True)`` metadata
and binds that metadata to the inference model's digest.

The published Ollama model declares ``gpt2/default``. Ollama's compatibility
loader changes that model's pre-tokenizer to Tekken, rather than GPT-2:
https://github.com/ollama/ollama/blob/main/llama/compat/llama-ollama-compat.cpp
The regex below follows llama.cpp's Tekken branch, including its documented
difference from Mistral's original Unicode regex:
https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp

The already installed ``tokenizers`` library performs byte-level BPE using the
supplied vocabulary/merges and Tekken's whole-piece vocabulary lookup. Counts
exclude BOS and chat framing. The caller must reserve 512 framing tokens plus
the full output allowance and reconcile Ollama's reported prompt count. These
are measured estimates, not a promise of exact server tokenizer equivalence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import re
from typing import Any

from app.providers.ollama_token_budget import LocalTokenBudgetError, _byte_alphabet


_EXPECTED_VOCAB_SIZE = 131_072
_EXPECTED_MERGE_COUNT = 269_443
_SPECIAL_COUNT = 1_000

# Hashes of CRLF-normalized, outer-whitespace-stripped standard templates:
# https://ollama.com/library/ministral-3:8b/blobs/5b74c26b9e0b
# https://ollama.com/library/ministral-3:14b/blobs/aa65b5aebf89
# The inspected 14B template differs only in the default model identity string;
# all tokenizer metadata, vocabulary, merges and special tokens match 8B.
# Exact matching prevents custom templates from duplicating input or injecting
# content. An explicit system message disables its large dated default prompt.
_STANDARD_MINISTRAL_TEMPLATE_SHA256 = "ab2bb6a5d55d3857ed25790b2ea7f27e9a21e1478ffd6a37a7ca79dce38a13e1"
_STANDARD_MINISTRAL_14B_TEMPLATE_SHA256 = "4789ab48bdc47ce9a0ee0bd3801eff135a329b7d53398a348e901315309f0a63"

_TEKKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+"
    r"|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_TOKENIZER_FIELDS = {
    "model", "pre", "tokens", "scores", "token_type", "merges",
    "add_bos_token", "add_eos_token", "add_padding_token", "add_unknown_token",
    "bos_token_id", "eos_token_id", "padding_token_id", "unknown_token_id",
    "add_space_prefix", "remove_extra_whitespaces", "escape_whitespaces",
    "treat_whitespace_as_suffix",
}


def _special_tokens() -> tuple[str, ...]:
    tokens = [
        "<unk>", "<s>", "</s>", "[INST]", "[/INST]", "[AVAILABLE_TOOLS]",
        "[/AVAILABLE_TOOLS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]", "[TOOL_CALLS]",
        "[IMG]", "<pad>", "[IMG_BREAK]", "[IMG_END]", "[PREFIX]", "[MIDDLE]",
        "[SUFFIX]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[TOOL_CONTENT]",
    ] + [f"<SPECIAL_{i}>" for i in range(20, _SPECIAL_COUNT)]
    for index, token in {
        24: "[AUDIO]", 25: "[BEGIN_AUDIO]", 32: "[ARGS]", 33: "[CALL_ID]",
        34: "[THINK]", 35: "[/THINK]",
    }.items():
        tokens[index] = token
    return tuple(tokens)


def _reject() -> None:
    raise LocalTokenBudgetError(
        "Local Ministral counting requires complete validated Ministral 3 8B/14B "
        "vocabulary metadata and its standard template without hidden prompt content."
    )


def _validate_metadata(show_payload: Mapping[str, Any]) -> tuple[dict[str, int], list[tuple[str, str]]]:
    if not isinstance(show_payload, Mapping):
        _reject()
    info, details, template = (show_payload.get(k) for k in ("model_info", "details", "template"))
    if (
        not isinstance(info, Mapping) or not isinstance(details, Mapping)
        or details.get("family") != "mistral3" or details.get("format") != "gguf"
        or info.get("general.architecture") != "mistral3"
        or info.get("tokenizer.ggml.model") != "gpt2"
        or info.get("tokenizer.ggml.pre") not in ("default", "tekken")
        or type(info.get("mistral3.vocab_size")) is not int
        or info.get("mistral3.vocab_size") != _EXPECTED_VOCAB_SIZE
        or show_payload.get("system") not in (None, "")
        or show_payload.get("messages") not in (None, [])
        or not isinstance(template, str) or len(template) > 8_192
    ):
        _reject()
    # This marker is one of Ollama's documented compatibility detection keys.
    # A default-pretokenizer GGUF without it cannot safely be assumed Tekken.
    if info["tokenizer.ggml.pre"] == "default" and not any(
        key in info for key in (
            "mistral3.rope.scaling.beta_fast", "mistral3.rope.scaling.mscale_all_dim",
            "mistral3.rope.scaling_beta",
        )
    ):
        _reject()
    try:
        template_hash = hashlib.sha256(template.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()
    except UnicodeError:
        _reject()
    if template_hash not in (_STANDARD_MINISTRAL_TEMPLATE_SHA256, _STANDARD_MINISTRAL_14B_TEMPLATE_SHA256):
        _reject()
    for key in info:
        if not isinstance(key, str):
            _reject()
        if key.startswith("tokenizer.") and (
            not key.startswith("tokenizer.ggml.")
            or key.removeprefix("tokenizer.ggml.") not in _TOKENIZER_FIELDS
        ):
            _reject()
    for field, expected in (
        ("add_bos_token", True), ("add_eos_token", False),
        ("add_padding_token", False), ("add_unknown_token", False),
    ):
        if info.get(f"tokenizer.ggml.{field}") is not expected:
            _reject()
    for field in ("add_space_prefix", "remove_extra_whitespaces", "escape_whitespaces", "treat_whitespace_as_suffix"):
        key = f"tokenizer.ggml.{field}"
        if key in info and info[key] is not False:
            _reject()
    for field, expected in (("bos_token_id", 1), ("eos_token_id", 2), ("padding_token_id", 11), ("unknown_token_id", 0)):
        value = info.get(f"tokenizer.ggml.{field}")
        if type(value) is not int or value != expected:
            _reject()

    tokens, scores, kinds, merges = (info.get(f"tokenizer.ggml.{key}") for key in ("tokens", "scores", "token_type", "merges"))
    if (
        not isinstance(tokens, list) or len(tokens) != _EXPECTED_VOCAB_SIZE
        or any(not isinstance(token, str) or not token or len(token) > 4_096 for token in tokens)
        or not isinstance(scores, list) or len(scores) != len(tokens)
        or any(type(score) not in (int, float) or score != i for i, score in enumerate(scores))
        or not isinstance(kinds, list) or len(kinds) != len(tokens)
        or any(type(kind) is not int or kind != (3 if i < _SPECIAL_COUNT else 1) for i, kind in enumerate(kinds))
        or tuple(tokens[:_SPECIAL_COUNT]) != _special_tokens()
        or not isinstance(merges, list) or len(merges) != _EXPECTED_MERGE_COUNT
    ):
        _reject()
    try:
        if sum(len(token.encode("utf-8")) for token in tokens) > 16 * 1024 * 1024:
            _reject()
    except UnicodeError:
        _reject()
    vocab = {token: index for index, token in enumerate(tokens)}
    if len(vocab) != len(tokens) or not _byte_alphabet().issubset(tokens[_SPECIAL_COUNT:]):
        _reject()
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    merge_bytes = 0
    for merge in merges:
        if not isinstance(merge, str) or not 3 <= len(merge) <= 8_193 or merge in seen:
            _reject()
        pieces = merge.split(" ")
        if len(pieces) != 2 or any(vocab.get(piece, -1) < _SPECIAL_COUNT for piece in (*pieces, "".join(pieces))):
            _reject()
        merge_bytes += len(merge.encode("utf-8"))
        if merge_bytes > 32 * 1024 * 1024:
            _reject()
        pairs.append((pieces[0], pieces[1]))
        seen.add(merge)
    return vocab, pairs


def _build_tokenizer(vocab: dict[str, int], merges: list[tuple[str, str]]) -> Any:
    # Construct entirely in memory. from_pretrained/from_file are never used.
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(_TEKKEN_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def build_ministral_token_counter(show_payload: Mapping[str, Any]) -> Callable[[list[dict[str, str]]], int]:
    """Build a fail-closed explicit-system/user text counter for Ministral 3.

    Single user requests are rejected because they activate the standard
    template's implicit dated system prompt. Tools, images, history and literal
    model control tokens are likewise unsupported. Inputs are never modified.
    """
    vocab, pairs = _validate_metadata(show_payload)
    vocabulary_size = len(vocab)
    try:
        tokenizer = _build_tokenizer(vocab, pairs)
        if tokenizer.get_vocab() != vocab:
            raise ValueError("vocabulary reindexed")
        reserved = re.compile("|".join(re.escape(token) for token in _special_tokens()))
    except Exception:
        raise LocalTokenBudgetError(
            "Unable to construct the local Ministral tokenizer using installed tokenizers; "
            "no inference or download was requested."
        ) from None

    def count(messages: list[dict[str, str]]) -> int:
        if (
            not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(message, dict) or set(message) != {"role", "content"} for message in messages)
            or [message["role"] for message in messages] != ["system", "user"]
        ):
            raise LocalTokenBudgetError(
                "Local Ministral counting requires exactly one explicit system and one user message."
            )
        if any(
            not isinstance(message["content"], str) or not message["content"].strip()
            or len(message["content"]) > 1_000_000 or reserved.search(message["content"])
            for message in messages
        ):
            raise LocalTokenBudgetError(
                "Local Ministral counting requires bounded nonempty text without reserved model tokens."
            )
        try:
            total = 0
            for message in messages:
                content = message["content"]
                byte_size = len(content.encode("utf-8"))
                ids = tokenizer.encode(content, add_special_tokens=False).ids
                if (
                    not isinstance(ids, list) or not 0 < len(ids) <= byte_size
                    or any(type(index) is not int or not _SPECIAL_COUNT <= index < vocabulary_size for index in ids)
                    or tokenizer.decode(ids, skip_special_tokens=False) != content
                ):
                    raise ValueError("invalid tokenization or incomplete text round-trip")
                total += len(ids)
            return total
        except Exception:
            raise LocalTokenBudgetError(
                "Local Ministral text token counting failed; no input was removed or sent."
            ) from None

    return count
