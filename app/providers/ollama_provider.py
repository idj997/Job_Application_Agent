"""Local Ollama inference with complete, validated outputs and no cloud fallback."""

from __future__ import annotations

import ipaddress
import json
import math
import os
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ValidationError

from app.providers.base import ModelProviderError

T = TypeVar("T", bound=BaseModel)
_TEMPLATE_RESERVE = 512
# Content is tokenized separately from chat framing. Allow a small discrepancy
# at its two edges: two tokens per edge, four per message. Public request paths
# emit one user message or system+user, so this tolerance is at most eight tokens.
# This is a fail-closed reconciliation policy, not a mathematical tokenizer bound
# or proof that the server retained the semantic content of the complete input.
_CACHED_BOUNDARY_TOLERANCE_PER_MESSAGE = 4


class OllamaProviderError(ModelProviderError):
    """A local inference error whose message does not expose candidate content."""


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _integer_setting(value: int | None, name: str, default: int, low: int, high: int) -> int:
    if value is None:
        try:
            value = int(os.getenv(name) or str(default))
        except ValueError:
            raise ValueError(f"{name} must be an integer between {low} and {high}.") from None
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}.")
    return value


def _local_host(value: str) -> str:
    """Accept only literal loopback hosts; normalize localhost without using DNS."""
    if not isinstance(value, str) or any(character.isspace() for character in value) or "\\" in value:
        raise ValueError("OLLAMA_BASE_URL must be a local HTTP(S) loopback URL.")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise ValueError("OLLAMA_BASE_URL has an invalid host or port.") from None
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.path not in {"", "/"} or parts.query or parts.fragment):
        raise ValueError("OLLAMA_BASE_URL must be a local HTTP(S) URL without credentials, paths or queries.")
    hostname = parts.hostname.casefold().rstrip(".")
    if hostname == "localhost":
        hostname = "127.0.0.1"
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        raise ValueError("OLLAMA_BASE_URL must use localhost or a literal loopback IP; remote hosts are disabled.") from None
    if not address.is_loopback or "%" in hostname or port == 0:
        raise ValueError("OLLAMA_BASE_URL must use a loopback IP and a valid local port; remote hosts are disabled.")
    hostname = f"[{address}]" if address.version == 6 else str(address)
    authority = hostname + (f":{port}" if port is not None else "")
    return urlunsplit((parts.scheme, authority, "", "", ""))


class OllamaProvider:
    """Qwen-compatible local provider with the existing model/host arguments.

    Context checks use UTF-8 byte length as a conservative upper estimate of
    Qwen's byte-tokenized text, plus 512 tokens for its chat template, reserving
    the entire output allowance. This is not a tokenizer and can overestimate
    English text substantially. Increase OLLAMA_NUM_CTX explicitly when needed;
    no prompt, CV evidence or job requirements are silently removed.

    Ollama's chat API has no documented truncate=False or tokenization endpoint.
    This estimate assumes a normal Qwen chat template; a custom Modelfile with
    large hidden system content needs additional context reserved by its owner.

    An opt-in input_token_counter can use the installed model's verified cached
    vocabulary instead. Its caller must bind metadata to the actual model; the
    template reserve, full-output reservation and reported-token reconciliation
    still apply. keep_alive=0 releases the model after each request, useful for
    sequential memory-limited drafting/audit trials. Neither option is enabled
    implicitly by environment loading or classifier matching.
    """

    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        *,
        client: Any | None = None,
        timeout: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        input_token_counter: Callable[[list[dict[str, str]]], int] | None = None,
        keep_alive: int | None = None,
        system_message_mode: str = "separate",
        think: bool | None = False,
    ) -> None:
        _load_environment()
        selected_model = model if model is not None else (os.getenv("OLLAMA_MODEL") or "qwen3:8b")
        if not isinstance(selected_model, str) or not selected_model.strip():
            raise ValueError("OLLAMA_MODEL must identify an installed local model, for example qwen3:8b.")
        self.model = selected_model.strip()
        if self.model.casefold().endswith((":cloud", "-cloud")) or any(character.isspace() for character in self.model):
            raise ValueError("Use an installed local Ollama model; cloud model tags are disabled.")
        self.host = _local_host(host if host is not None else (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"))
        if timeout is None:
            try:
                timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS") or "600")
            except ValueError:
                raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than 0 and at most 7200 seconds.") from None
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 7200:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than 0 and at most 7200 seconds.")
        self.timeout = float(timeout)
        self.num_ctx = _integer_setting(num_ctx, "OLLAMA_NUM_CTX", 32768, 1024, 131072)
        self.num_predict = _integer_setting(num_predict, "OLLAMA_NUM_PREDICT", 4096, 1, 32768)
        if self.num_predict + _TEMPLATE_RESERVE >= self.num_ctx:
            raise ValueError("OLLAMA_NUM_CTX must leave room for the prompt as well as OLLAMA_NUM_PREDICT and the template reserve.")
        if input_token_counter is not None and not callable(input_token_counter):
            raise ValueError("input_token_counter must be a verified local token-counting callable.")
        if keep_alive is not None and (type(keep_alive) is not int or not 0 <= keep_alive <= 3600):
            raise ValueError("keep_alive must be an integer number of seconds between 0 and 3600.")
        self.input_token_counter = input_token_counter
        self.keep_alive = keep_alive
        if system_message_mode not in {"separate", "merge_user"}:
            raise ValueError("system_message_mode must be separate or merge_user.")
        if think is not None and type(think) is not bool:
            raise ValueError("think must be a boolean or None to omit the option.")
        self.system_message_mode = system_message_mode
        self.think = think
        self.budget_diagnostics: list[dict] = []
        self._usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if client is not None:
            self.client = client
            return
        try:
            from ollama import Client
        except ImportError:
            raise ImportError("Install the local inference SDK with: pip install ollama") from None
        self.client = Client(
            host=self.host, timeout=self.timeout, follow_redirects=False, trust_env=False,
            # The SDK otherwise inherits OLLAMA_API_KEY. Local inference does not
            # need that credential; a fixed inert value prevents forwarding it.
            headers={"Authorization": "Bearer local-only"},
        )

    @property
    def usage(self) -> dict[str, int]:
        """Aggregate local calls and token counts reported by Ollama."""
        return self._usage.copy()

    def generate(self, prompt: str) -> str:
        return self._request([{"role": "user", "content": prompt}])

    def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None,
    ) -> T:
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("schema must be a Pydantic BaseModel class.")
        output_schema = schema.model_json_schema()
        instructions = (
            "Return exactly one JSON object matching the following schema. "
            "Do not wrap it in Markdown or add commentary.\n"
            + json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
        )
        if system_prompt:
            instructions = system_prompt + "\n\n" + instructions
        messages = ([{"role": "user", "content": instructions + "\n\n" + prompt}]
                    if self.system_message_mode == "merge_user" else
                    [{"role": "system", "content": instructions}, {"role": "user", "content": prompt}])
        content = self._request(messages, output_schema=output_schema)
        try:
            # Reject fences, explanatory text, partial JSON and schema violations.
            return schema.model_validate_json(content)
        except ValidationError:
            raise OllamaProviderError(
                "Local Ollama output was not valid JSON for the requested schema. "
                "Try a fresh local preparation; no partial result was accepted."
            ) from None

    def _request(self, messages: list[dict[str, str]], *, output_schema: dict | None = None) -> str:
        if any(not isinstance(message["content"], str) or not message["content"].strip() for message in messages):
            raise ValueError("Ollama prompts must contain non-empty text.")
        counted_text_tokens = None
        if self.input_token_counter is None:
            input_bound = sum(len(message["content"].encode("utf-8")) for message in messages) + _TEMPLATE_RESERVE
            method = "UTF-8 bytes (upper estimate)"
        else:
            try:
                count = self.input_token_counter(messages)
            except Exception:
                raise OllamaProviderError("The cached local tokenizer could not count the complete prompt; no request was sent.") from None
            if type(count) is not int or count <= 0:
                raise OllamaProviderError("The cached local tokenizer returned an invalid prompt count; no request was sent.")
            counted_text_tokens = count
            input_bound = counted_text_tokens + _TEMPLATE_RESERVE
            method = "cached model vocabulary estimate plus chat-template reserve"
        if input_bound + self.num_predict > self.num_ctx:
            raise OllamaProviderError(
                f"Local prompt exceeds the conservative context allowance: {input_bound} "
                f"estimated input tokens plus {self.num_predict} reserved output tokens, "
                f"but OLLAMA_NUM_CTX is {self.num_ctx}. Counting method: {method}. "
                "Increase OLLAMA_NUM_CTX within your "
                "model/memory limits or reduce OLLAMA_NUM_PREDICT; no input was truncated."
            )
        self._usage["requests"] += 1
        options = {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": self.num_predict}
        parameters = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if self.think is not None:
            parameters["think"] = self.think
        if self.keep_alive is not None:
            parameters["keep_alive"] = self.keep_alive
        if output_schema is not None:
            parameters["format"] = output_schema
        try:
            response = self.client.chat(**parameters)
        except Exception as error:
            status = _field(error, "status_code")
            if status == 404:
                detail = "The local model was not found. Check ollama list and OLLAMA_MODEL."
            elif type(error).__name__ in {"ReadTimeout", "ConnectTimeout", "TimeoutException"}:
                detail = "Local inference timed out. Check Ollama/GPU availability or increase OLLAMA_TIMEOUT_SECONDS."
            elif type(status) is int and 300 <= status < 400:
                detail = "The local endpoint redirected the request; redirects are disabled."
            else:
                detail = "Check that the local Ollama server is running and the model fits available memory."
            raise OllamaProviderError(f"Local Ollama request failed. {detail}") from None
        for remote, local in (("prompt_eval_count", "input_tokens"), ("eval_count", "output_tokens")):
            count = _field(response, remote, 0)
            if type(count) is int and count >= 0:
                self._usage[local] += count
        self._usage["total_tokens"] = self._usage["input_tokens"] + self._usage["output_tokens"]
        if _field(response, "error"):
            raise OllamaProviderError("Local Ollama returned an error without a complete result. Check the local server.")
        if _field(response, "done") is not True:
            raise OllamaProviderError("Local Ollama returned an incomplete response; no partial output was accepted.")
        reason = _field(response, "done_reason")
        if reason == "length":
            raise OllamaProviderError(
                "Local Ollama reached its generation limit. Increase OLLAMA_NUM_PREDICT "
                "and OLLAMA_NUM_CTX if needed; truncated output was rejected."
            )
        if reason != "stop":
            raise OllamaProviderError("Local Ollama did not report a normal completion; no output was accepted.")
        prompt_count = _field(response, "prompt_eval_count", 0)
        self.budget_diagnostics.append({
            "method": method, "estimated_input_with_reserve": input_bound,
            "reported_prompt_tokens": prompt_count if type(prompt_count) is int else None,
            "reserved_output_tokens": self.num_predict, "context_tokens": self.num_ctx,
        })
        if counted_text_tokens is not None:
            boundary_tolerance = _CACHED_BOUNDARY_TOLERANCE_PER_MESSAGE * min(len(messages), 2)
            minimum_reported = max(1, counted_text_tokens - boundary_tolerance)
            self.budget_diagnostics[-1].update(
                counted_text_tokens=counted_text_tokens,
                boundary_tolerance_tokens=boundary_tolerance,
                minimum_reported_prompt_tokens=minimum_reported,
            )
            if (type(prompt_count) is not int or prompt_count < minimum_reported
                    or prompt_count > input_bound):
                raise OllamaProviderError(
                    "Ollama's reported prompt count could not be reconciled with the cached-tokenizer budget; "
                    "the output was rejected and requires a context-budget review."
                )
        if type(prompt_count) is int and prompt_count + self.num_predict > self.num_ctx:
            raise OllamaProviderError(
                "Local Ollama reported a prompt too close to the context limit. "
                "Increase OLLAMA_NUM_CTX; the result was rejected to avoid context loss."
            )
        message = _field(response, "message")
        if _field(message, "tool_calls"):
            raise OllamaProviderError("Local Ollama returned a tool request instead of a plain result; no tools were executed.")
        content = _field(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaProviderError("Local Ollama returned no final answer text.")
        return content
