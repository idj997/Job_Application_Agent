"""Opt-in text-token estimates from an already installed Qwen GGUF vocabulary.

This module performs no network requests, file reads or model loading. Its caller
must obtain ``show_payload`` from the same local model used for inference and
ensure the model tag/digest does not change between inspection and the request.
The show response itself does not authenticate a model digest.

The returned count excludes chat framing: the provider must retain its 512-token
template reserve, reserve the full output allowance, and check Ollama's reported
prompt count after inference. This is a measured estimate, not an independently
verified exact reproduction of Ollama's tokenizer/template execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class LocalTokenBudgetError(ValueError):
    """A sanitized, fail-closed local tokenizer configuration error."""


# Only this inspected standard Qwen3 template is supported. A length check alone
# would not exclude a small custom template that repeats user content many times.
_STANDARD_QWEN3_TEMPLATE = '''
{{- $lastUserIdx := -1 -}}
{{- range $idx, $msg := .Messages -}}
{{- if eq $msg.Role "user" }}{{ $lastUserIdx = $idx }}{{ end -}}
{{- end }}
{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}
{{- if and $.IsThinkSet (eq $i $lastUserIdx) }}
   {{- if $.Think -}}
      {{- " "}}/think
   {{- else -}}
      {{- " "}}/no_think
   {{- end -}}
{{- end }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if (and $.IsThinkSet (and .Thinking (or $last (gt $i $lastUserIdx)))) -}}
<think>{{ .Thinking }}</think>
{{ end -}}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ if and $.IsThinkSet (not $.Think) -}}
<think>

</think>

{{ end -}}
{{ end }}
{{- end }}
'''


def _tokenizer_class() -> Any:
    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    except ImportError:
        raise LocalTokenBudgetError(
            "Local Qwen token counting needs the installed transformers tokenizer; "
            "no tokenizer or model was downloaded."
        ) from None
    return Qwen2Tokenizer


def _reject() -> None:
    raise LocalTokenBudgetError(
        "Local token counting requires validated Qwen3/qwen2 vocabulary metadata "
        "and the supported standard template without hidden system or message content."
    )


def _byte_alphabet() -> set[str]:
    """GPT-2's reversible byte alphabet, without importing/loading a model."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    return {chr(i) for i in visible} | {chr(256 + i) for i in range(256 - len(visible))}


def build_qwen_token_counter(
    show_payload: Mapping[str, Any],
) -> Callable[[list[dict[str, str]]], int]:
    """Build a bounded text counter from local ``show(verbose=True)`` metadata.

    Supports one user message or one system message followed by one user message;
    tools, conversation history and customized/stored prompt content are rejected.
    The input payload and message content are never logged or modified.
    """
    if not isinstance(show_payload, Mapping):
        _reject()
    info = show_payload.get("model_info")
    details = show_payload.get("details")
    template = show_payload.get("template")
    if (
        not isinstance(info, Mapping)
        or not isinstance(details, Mapping)
        or details.get("family") != "qwen3"
        or details.get("format") != "gguf"
        or info.get("general.architecture") != "qwen3"
        or info.get("tokenizer.ggml.pre") != "qwen2"
        or info.get("tokenizer.ggml.model") != "gpt2"
        or info.get("tokenizer.ggml.add_bos_token") is not False
        or show_payload.get("system") not in (None, "")
        or show_payload.get("messages") not in (None, [])
        or not isinstance(template, str)
        or template.replace("\r\n", "\n").strip() != _STANDARD_QWEN3_TEMPLATE.strip()
    ):
        _reject()

    tokens = info.get("tokenizer.ggml.tokens")
    token_types = info.get("tokenizer.ggml.token_type")
    merges = info.get("tokenizer.ggml.merges")
    if (
        not isinstance(tokens, list) or not 256 <= len(tokens) <= 262_144
        or not isinstance(token_types, list) or len(token_types) != len(tokens)
        or not isinstance(merges, list) or not 1 <= len(merges) <= 300_000
        or any(not isinstance(t, str) or not t or len(t) > 4096 for t in tokens)
        or any(type(t) is not int or t not in {1, 3, 4, 5} for t in token_types)
    ):
        _reject()
    if sum(len(t.encode("utf-8")) for t in tokens) > 16 * 1024 * 1024:
        _reject()
    vocab = {token: index for index, token in enumerate(tokens)}
    # Without all base bytes a tokenizer with no unknown-token fallback could
    # silently omit an otherwise valid input character and underestimate it.
    if len(vocab) != len(tokens) or not _byte_alphabet().issubset(vocab):
        _reject()
    for name, expected in (
        ("bos_token_id", "<|endoftext|>"),
        ("padding_token_id", "<|endoftext|>"),
        ("eos_token_id", "<|im_end|>"),
    ):
        index = info.get(f"tokenizer.ggml.{name}")
        if (type(index) is not int or not 0 <= index < len(tokens)
                or tokens[index] != expected or token_types[index] != 3):
            _reject()
    if "<|im_start|>" not in vocab or token_types[vocab["<|im_start|>"]] != 3:
        _reject()

    merge_pairs: list[tuple[str, str]] = []
    for merge in merges:
        if not isinstance(merge, str) or len(merge) > 8193:
            _reject()
        pieces = merge.split(" ")
        if (len(pieces) != 2 or any(piece not in vocab for piece in pieces)
                or "".join(pieces) not in vocab):
            _reject()
        merge_pairs.append((pieces[0], pieces[1]))
    if len(set(merge_pairs)) != len(merge_pairs):
        _reject()
    specials = [token for token, kind in zip(tokens, token_types) if kind in {3, 4}]
    if len(specials) > 128 or any(len(token) > 256 for token in specials):
        _reject()
    try:
        tokenizer = _tokenizer_class()(
            vocab=vocab,
            merges=merge_pairs,
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
            additional_special_tokens=specials,
        )
        # Catch constructors that ignore/reindex the supplied vocabulary.
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            if tokenizer.encode(token, add_special_tokens=False) != [vocab[token]]:
                _reject()
    except Exception:
        raise LocalTokenBudgetError(
            "Unable to construct the local Qwen tokenizer from the supplied metadata; "
            "no inference was requested."
        ) from None

    def count(messages: list[dict[str, str]]) -> int:
        if (not isinstance(messages, list) or len(messages) not in {1, 2}
                or any(not isinstance(m, dict) or set(m) != {"role", "content"}
                       for m in messages)):
            raise LocalTokenBudgetError("Local token counting supports only a plain system/user request.")
        roles = [m["role"] for m in messages]
        if roles not in (["user"], ["system", "user"]):
            raise LocalTokenBudgetError("Local token counting supports only a plain system/user request.")
        if any(not isinstance(m["content"], str) or not m["content"].strip()
               or len(m["content"]) > 1_000_000 for m in messages):
            raise LocalTokenBudgetError("Local token counting requires bounded nonempty text.")
        try:
            total = 0
            for message in messages:
                content = message["content"]
                ids = tokenizer.encode(content, add_special_tokens=False)
                # A byte-tokenized string cannot need more tokens than UTF-8
                # bytes; an empty result would hide input rather than count it.
                if (not isinstance(ids, list) or not ids
                        or len(ids) > len(content.encode("utf-8"))
                        or any(type(i) is not int or not 0 <= i < len(tokens) for i in ids)):
                    raise ValueError("invalid tokenizer result")
                total += len(ids)
            return total
        except Exception:
            raise LocalTokenBudgetError(
                "Local Qwen text token counting failed; no input was removed or sent."
            ) from None

    return count
