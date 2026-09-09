"""OpenAI Responses API adapter; importing it never initializes an API client."""

from __future__ import annotations

import math
import os
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class OpenAIProviderError(RuntimeError):
    """A request failed or returned no usable, complete result."""


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class OpenAIProvider:
    """Use a configured model with Responses and strict Pydantic outputs.

    ``client`` is injectable for offline tests. A real client is created lazily
    at construction, after configuration validation. ``usage`` counts logical
    requests (not internal SDK retries), and tokens reported by responses,
    including responses subsequently rejected as incomplete or refused.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_output_tokens: int | None = None,
    ) -> None:
        _load_environment()
        self.model = (model if model is not None else os.getenv("OPENAI_MODEL", "")).strip()
        if not self.model:
            raise ValueError("Set OPENAI_MODEL or pass model= with a Responses/Structured Outputs model.")
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("timeout must be greater than 0 and at most 300 seconds.")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer between 0 and 5.")
        if max_output_tokens is None:
            try:
                max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "6000"))
            except ValueError:
                raise ValueError("OPENAI_MAX_OUTPUT_TOKENS must be a positive integer.") from None
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 128_000:
            raise ValueError("max_output_tokens must be an integer between 1 and 128000.")

        self.max_output_tokens = max_output_tokens
        self._usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if client is not None:
            self.client = client
            return

        configured_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        if not configured_key.strip():
            raise ValueError("Set OPENAI_API_KEY to use the OpenAI application mode.")
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install the OpenAI SDK with: pip install -r requirements-openai.txt") from None
        self.client = OpenAI(api_key=configured_key, timeout=timeout, max_retries=max_retries)

    @property
    def usage(self) -> dict[str, int]:
        """Return reported token totals; unknown usage on failed calls is omitted."""
        return self._usage.copy()

    def generate(self, prompt: str) -> str:
        response = self._request("create", [{"role": "user", "content": prompt}])
        content = _field(response, "output_text")
        if not isinstance(content, str) or not content.strip():
            raise OpenAIProviderError("OpenAI returned no text output.")
        return content

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("schema must be a Pydantic BaseModel class.")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._request("parse", messages, text_format=schema)
        parsed = _field(response, "output_parsed")
        if parsed is None:
            raise OpenAIProviderError("OpenAI returned no parsed structured output. Check model support for Structured Outputs.")
        try:
            return schema.model_validate(parsed)
        except ValidationError:
            raise OpenAIProviderError("OpenAI output failed the requested schema validation.") from None

    def _request(self, method: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self._usage["requests"] += 1
        try:
            response = getattr(self.client.responses, method)(
                model=self.model,
                input=messages,
                max_output_tokens=self.max_output_tokens,
                store=False,
                **kwargs,
            )
        except Exception as error:
            # SDK exceptions may contain request bodies. Do not expose CVs,
            # prompts, credentials or raw response content in CLI errors.
            status = getattr(error, "status_code", None)
            if status in (401, 403):
                advice = "Check OPENAI_API_KEY and project/model access."
            elif status == 429:
                advice = "Check API quota/rate limits and retry later."
            elif status in (400, 404):
                advice = "Check OPENAI_MODEL, Structured Outputs support and the output token limit."
            elif type(error).__name__ == "LengthFinishReasonError":
                advice = "Increase OPENAI_MAX_OUTPUT_TOKENS or reduce the input size."
            else:
                advice = "Check API connectivity and configuration, then retry."
            detail = f" (HTTP {status})" if type(status) is int else ""
            raise OpenAIProviderError(f"OpenAI request failed{detail}. {advice}") from None

        usage = _field(response, "usage")
        for name in ("input_tokens", "output_tokens"):
            count = _field(usage, name, 0)
            if type(count) is int and count >= 0:
                self._usage[name] += count
        self._usage["total_tokens"] = self._usage["input_tokens"] + self._usage["output_tokens"]
        self._check_response(response)
        return response

    @staticmethod
    def _check_response(response: Any) -> None:
        status = _field(response, "status")
        if status == "incomplete":
            reason = _field(_field(response, "incomplete_details"), "reason")
            advice = (
                "Increase OPENAI_MAX_OUTPUT_TOKENS or reduce the input size."
                if reason == "max_output_tokens"
                else "Review the input and API response settings."
            )
            raise OpenAIProviderError(f"OpenAI response was incomplete. {advice}")
        if status != "completed":
            raise OpenAIProviderError("OpenAI did not return a completed response.")
        for output in _field(response, "output", []) or []:
            for content in _field(output, "content", []) or []:
                if _field(content, "type") == "refusal":
                    raise OpenAIProviderError("OpenAI refused the request; no usable result was produced.")
