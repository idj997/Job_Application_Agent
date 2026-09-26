"""Sequential, digest-bound local writer/critic runtimes; no automatic downloads."""
from __future__ import annotations

import math
import os
from typing import Any

from app.providers.ollama_provider import OllamaProvider, OllamaProviderError, _field, _local_host


def local_role_config(model: str, role: str) -> dict:
    """Pure configuration parsing, suitable for dry-run/cache identity."""
    if role not in {"writer", "critic"}:
        raise ValueError("Local model role must be writer or critic.")
    if (not isinstance(model, str) or not model.strip() or any(c.isspace() for c in model)
            or model.casefold().endswith((":cloud", "-cloud"))):
        raise ValueError("A local writer/critic requires an installed local model tag.")
    prefix = "OLLAMA" if role == "writer" else "CV_CRITIC"

    def integer(suffix, default, low, high):
        try:
            value = int(os.getenv(prefix + "_" + suffix) or default)
        except (TypeError, ValueError):
            raise ValueError(f"{prefix}_{suffix} must be an integer between {low} and {high}.") from None
        if not low <= value <= high:
            raise ValueError(f"{prefix}_{suffix} must be an integer between {low} and {high}.")
        return value

    context = integer("NUM_CTX", 12288, 1024, 131072)
    output = integer("NUM_PREDICT", 4096, 1, 32768)
    try:
        timeout = float(os.getenv(prefix + "_TIMEOUT_SECONDS") or "900")
    except ValueError:
        raise ValueError(f"{prefix}_TIMEOUT_SECONDS must be between 0 and 7200 seconds.") from None
    if not math.isfinite(timeout) or not 0 < timeout <= 7200:
        raise ValueError(f"{prefix}_TIMEOUT_SECONDS must be between 0 and 7200 seconds.")
    if output + 512 >= context:
        raise ValueError(f"{prefix}_NUM_CTX must leave room for the prompt and reserved output.")
    return {"provider": "ollama", "model": model, "role": role,
            "host": _local_host(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"),
            "num_ctx": context, "num_predict": output, "timeout_seconds": timeout,
            "keep_alive": 0, "cpu_threads": 6, "batch_size": 128, "protocol_version": 1}


class LocalRoleProvider:
    """Lazily inspect cached vocabulary; bind every call to that model's digest.

    Each successful response requests immediate unloading, so the workflow can
    alternate models without deliberately keeping both resident on an 8 GB GPU.
    No inference, model loading, or network request happens during construction.
    """
    name = "ollama"

    def __init__(self, model: str, role: str, *, client: Any | None = None):
        self.fingerprint = local_role_config(model, role)
        self.model = model
        self.model_digest = None
        self.num_ctx = self.fingerprint["num_ctx"]
        self.num_predict = self.fingerprint["num_predict"]
        self.timeout = self.fingerprint["timeout_seconds"]
        self._client = client
        self._provider = None

    @property
    def usage(self):
        return self._provider.usage if self._provider else {
            "requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        }

    @property
    def budget_diagnostics(self):
        return self._provider.budget_diagnostics if self._provider else []

    def _digest(self, client):
        try:
            models = _field(client.list(), "models", [])
            # Ollama normalizes bare tags to :latest; use the same exact alias.
            names = {self.model, self.model + ":latest"} if ":" not in self.model else {self.model}
            matches = [item for item in models if (_field(item, "model") or _field(item, "name")) in names]
            digest = _field(matches[0], "digest") if len(matches) == 1 else None
            if not isinstance(digest, str) or not digest:
                raise ValueError
            return digest
        except Exception:
            raise OllamaProviderError("Could not verify the installed local model digest; no fallback or download was attempted.") from None

    def _load(self):
        if self._provider is not None:
            return self._provider
        provider = OllamaProvider(model=self.model, host=self.fingerprint["host"],
                                  num_ctx=self.num_ctx, num_predict=self.num_predict,
                                  timeout=self.timeout, keep_alive=0, client=self._client)
        client = provider.client
        digest = self._digest(client)
        try:
            if callable(getattr(client, "_request_raw", None)):
                # Older installed Ollama SDKs omit the server's verbose option
                # from show(). Reuse their configured, loopback-only transport.
                # Keep raw server fields: older ShowResponse also discards
                # system/messages, which must be checked for hidden content.
                response = client._request_raw("POST", "/api/show",
                    json={"model": self.model, "verbose": True}).json()
            else:
                response = client.show(self.model, verbose=True)
            shown = response.model_dump() if hasattr(response, "model_dump") else response
            family = shown.get("details", {}).get("family")
            if family == "qwen3":
                from app.providers.ollama_token_budget import build_qwen_token_counter
                counter = build_qwen_token_counter(shown)
            elif family == "gemma3":
                from app.providers.gemma_token_budget import build_gemma_token_counter
                counter = build_gemma_token_counter(shown)
                provider.system_message_mode = "merge_user"
                provider.think = None
            elif family == "mistral3":
                from app.providers.ministral_token_budget import build_ministral_token_counter
                counter = build_ministral_token_counter(shown)
                # An explicit system message disables the standard template's
                # hidden Le Chat/date preamble; the counter enforces this shape.
                provider.system_message_mode = "separate"
                provider.think = None
            else:
                raise ValueError("Unsupported cached tokenizer family")
        except Exception:
            raise OllamaProviderError(
                "Could not validate the cached Qwen3/Gemma3/Ministral3 tokenizer and standard template. "
                "No inference, download, truncation or fallback was attempted."
            ) from None
        if self._digest(client) != digest:
            raise OllamaProviderError("The local model changed during tokenizer inspection; no inference was attempted.")
        self.model_digest = digest
        provider.input_token_counter = counter
        owner = self

        class BoundClient:
            def chat(self, **kwargs):
                if kwargs.get("model") != owner.model or owner._digest(client) != owner.model_digest:
                    raise OllamaProviderError("The local model changed after tokenizer inspection.")
                kwargs["options"].update(num_thread=6, num_batch=128)
                result = client.chat(**kwargs)
                if owner._digest(client) != owner.model_digest:
                    raise OllamaProviderError("The local model changed during inference; output was rejected.")
                return result

        provider.client = BoundClient()
        self._provider = provider
        return provider

    def generate(self, prompt):
        return self._load().generate(prompt)

    def generate_structured(self, prompt, schema, system_prompt=None):
        try:
            return self._load().generate_structured(prompt=prompt, schema=schema, system_prompt=system_prompt)
        except OllamaProviderError as exc:
            message = str(exc)
            if self.fingerprint["role"] == "critic":
                for suffix in ("MODEL", "NUM_CTX", "NUM_PREDICT", "TIMEOUT_SECONDS"):
                    message = message.replace("OLLAMA_" + suffix, "CV_CRITIC_" + suffix)
            raise OllamaProviderError(message) from None


def build_local_role_provider(model: str, role: str) -> LocalRoleProvider:
    return LocalRoleProvider(model, role)
