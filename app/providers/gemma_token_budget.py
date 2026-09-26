"""Opt-in text-token estimates from an installed Gemma 3 GGUF vocabulary.

No model loading, downloads, file access or requests occur in this module. The
caller supplies local ``show(verbose=True)`` metadata and must bind its digest to
the model actually used. Only the inspected Gemma 3 template and tokenizer are
supported; this is not a generic Gemma/Gemma 4 tokenizer.

Gemma 3's GGUF ``llama`` tokenizer uses score-priority pair merging. An Unigram
Viterbi converter can choose different segmentations for these rank-like scores,
so this counter follows the inspected llama.cpp SPM merge ordering instead:
https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp

Counts exclude BOS and chat framing. They are estimates, not proof of identical
server execution: reserve 512 framing tokens plus the entire output allowance,
and reconcile the server-reported prompt count before accepting its output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import heapq
import math
from typing import Any

from app.providers.ollama_token_budget import LocalTokenBudgetError


_EXPECTED_VOCAB_SIZE = 262_145
_EXPECTED_MERGE_COUNT = 514_906
_STANDARD_GEMMA3_TEMPLATE = '''{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}'''

_TOKENIZER_FIELDS = {
    "model", "pre", "tokens", "scores", "token_type", "merges",
    "add_bos_token", "add_eos_token", "add_padding_token", "add_unknown_token",
    "bos_token_id", "eos_token_id", "padding_token_id", "unknown_token_id",
    "add_space_prefix", "remove_extra_whitespaces", "escape_whitespaces",
    "treat_whitespace_as_suffix",
}


def _reject() -> None:
    raise LocalTokenBudgetError(
        "Local Gemma counting requires complete validated Gemma 3 vocabulary "
        "metadata and its standard template without hidden prompt content."
    )


def _validate_metadata(show_payload: Mapping[str, Any]) -> tuple[dict[str, int], tuple[float, ...]]:
    if not isinstance(show_payload, Mapping):
        _reject()
    info = show_payload.get("model_info")
    details = show_payload.get("details")
    template = show_payload.get("template")
    if (
        not isinstance(info, Mapping)
        or not isinstance(details, Mapping)
        or details.get("family") != "gemma3"
        or details.get("format") != "gguf"
        or info.get("general.architecture") != "gemma3"
        or info.get("tokenizer.ggml.model") != "llama"
        or info.get("tokenizer.ggml.pre") != "default"
        or show_payload.get("system") not in (None, "")
        or show_payload.get("messages") not in (None, [])
        or not isinstance(template, str)
        or template.replace("\r\n", "\n").strip() != _STANDARD_GEMMA3_TEMPLATE.strip()
    ):
        _reject()
    # Reject unknown normalization and special-token settings rather than
    # treating a custom tokenizer as the installed standard configuration.
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
    for field, expected in (
        ("add_space_prefix", True), ("remove_extra_whitespaces", False),
        ("escape_whitespaces", True), ("treat_whitespace_as_suffix", False),
    ):
        key = f"tokenizer.ggml.{field}"
        if key in info and info[key] is not expected:
            _reject()

    tokens = info.get("tokenizer.ggml.tokens")
    scores = info.get("tokenizer.ggml.scores")
    kinds = info.get("tokenizer.ggml.token_type")
    if (
        not isinstance(tokens, list) or len(tokens) != _EXPECTED_VOCAB_SIZE
        or not isinstance(scores, list) or len(scores) != len(tokens)
        or not isinstance(kinds, list) or len(kinds) != len(tokens)
        or any(not isinstance(t, str) or not t or len(t) > 4096 for t in tokens)
        or any(type(s) not in (int, float) or not -10_000_000 <= s <= 0
               or not math.isfinite(s) for s in scores)
        or any(type(k) is not int or k not in {1, 2, 3, 4, 6} for k in kinds)
    ):
        _reject()
    try:
        if sum(len(t.encode("utf-8")) for t in tokens) > 16 * 1024 * 1024:
            _reject()
    except UnicodeError:
        _reject()
    vocab = {token: index for index, token in enumerate(tokens)}
    if len(vocab) != len(tokens):
        _reject()
    expected_specials = {
        "<pad>": (0, 3), "<eos>": (1, 3), "<bos>": (2, 3),
        "<unk>": (3, 2), "<start_of_turn>": (105, 3),
        "<end_of_turn>": (106, 3), "<image_soft_token>": (len(tokens) - 1, 4),
    }
    actual_specials = {
        token: (i, kinds[i]) for i, token in enumerate(tokens) if kinds[i] in {2, 3, 4}
    }
    if actual_specials != expected_specials:
        _reject()
    for field, expected in (
        ("padding_token_id", 0), ("eos_token_id", 1),
        ("bos_token_id", 2), ("unknown_token_id", 3),
    ):
        value = info.get(f"tokenizer.ggml.{field}")
        if type(value) is not int or value != expected:
            _reject()
    if {token for token, kind in zip(tokens, kinds) if kind == 6} != {
        f"<0x{i:02X}>" for i in range(256)
    }:
        _reject()
    if "▁" not in vocab or kinds[vocab["▁"]] != 1:
        _reject()

    # The llama/SPM path uses scores, not this export's BPE merge table. Still
    # validate the table's completeness/consistency to reject damaged metadata.
    merges = info.get("tokenizer.ggml.merges")
    if not isinstance(merges, list) or len(merges) != _EXPECTED_MERGE_COUNT:
        _reject()
    seen: set[str] = set()
    merge_bytes = 0
    for merge in merges:
        if not isinstance(merge, str) or not 3 <= len(merge) <= 8193 or merge in seen:
            _reject()
        pieces = merge.split(" ")
        if (len(pieces) != 2 or any(piece not in vocab for piece in pieces)
                or "".join(pieces) not in vocab):
            _reject()
        merge_bytes += len(merge.encode("utf-8"))
        if merge_bytes > 32 * 1024 * 1024:
            _reject()
        seen.add(merge)
    return vocab, tuple(float(s) for s in scores)


def _count_spm_text(content: str, vocab: dict[str, int], scores: tuple[float, ...]) -> int:
    """Count score-priority merges, breaking equal scores by leftmost position.

    Only ASCII spaces are escaped. There is no NFKC normalization, whitespace
    collapse, added prefix or tab rewrite. BOS/prefix/chat-boundary behavior is
    accounted for by the caller's framing reserve and count reconciliation.
    """
    symbols = list(content.replace(" ", "▁"))
    previous = list(range(-1, len(symbols) - 1))
    following = list(range(1, len(symbols))) + [-1]
    queue: list[tuple[float, int, int, int]] = []

    def enqueue(left: int, right: int) -> None:
        if left < 0 or right < 0:
            return
        merged = symbols[left] + symbols[right]
        token_id = vocab.get(merged)
        if token_id is not None:
            heapq.heappush(queue, (-scores[token_id], left, right, len(merged)))

    for right in range(1, len(symbols)):
        enqueue(right - 1, right)
    while queue:
        _, left, right, size = heapq.heappop(queue)
        if (not symbols[left] or not symbols[right] or following[left] != right
                or len(symbols[left]) + len(symbols[right]) != size):
            continue
        symbols[left] += symbols[right]
        symbols[right] = ""
        following[left] = following[right]
        if following[right] >= 0:
            previous[following[right]] = left
        enqueue(previous[left], left)
        enqueue(left, following[left])

    count = 0
    position = 0
    while position != -1:
        symbol = symbols[position]
        # Every merged symbol is a vocabulary entry; only unmerged Unicode
        # characters can need byte fallback. All 256 byte entries were checked.
        count += 1 if symbol in vocab else len(symbol.encode("utf-8"))
        position = following[position]
    return count


def build_gemma_token_counter(show_payload: Mapping[str, Any]) -> Callable[[list[dict[str, str]]], int]:
    """Build a fail-closed single-user-message Gemma 3 text counter.

    Instructions must already be merged into the user message. Reserved control,
    unknown and image-token literals are rejected rather than approximating
    model-specific special-token partitioning inside untrusted CV/job content.
    """
    vocab, scores = _validate_metadata(show_payload)
    reserved = (
        "<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>",
        "<end_of_turn>", "<image_soft_token>",
    )

    def count(messages: list[dict[str, str]]) -> int:
        if (
            not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict)
            or set(messages[0]) != {"role", "content"}
            or messages[0]["role"] != "user"
        ):
            raise LocalTokenBudgetError("Local Gemma counting supports one plain user message only.")
        content = messages[0]["content"]
        if (not isinstance(content, str) or not content.strip() or len(content) > 1_000_000
                or any(token in content for token in reserved)):
            raise LocalTokenBudgetError(
                "Local Gemma counting requires bounded nonempty text without reserved model tokens."
            )
        try:
            encoded_size = len(content.encode("utf-8"))
            result = _count_spm_text(content, vocab, scores)
            if type(result) is not int or not 0 < result <= encoded_size:
                raise ValueError("invalid counter result")
            return result
        except Exception:
            raise LocalTokenBudgetError(
                "Local Gemma text token counting failed; no input was removed or sent."
            ) from None

    return count
