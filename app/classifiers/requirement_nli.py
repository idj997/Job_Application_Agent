"""Lazy, offline-first adapter for the existing pretrained NLI baseline."""

from __future__ import annotations

import math
import os
import re
from typing import Any

from app.providers.base import ModelProviderError


class NLIInputTooLong(ModelProviderError):
    """An entire sentence pair does not fit; no tokens were removed."""


class RequirementNLIClassifier:
    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        revision: str | None = None,
        device: str | None = None,
        local_files_only: bool = True,
        max_length: int = 256,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("REQUIREMENT_NLI_MODEL") or self.MODEL_NAME
        self.revision = revision or os.getenv("REQUIREMENT_NLI_REVISION") or None
        self.device = device or os.getenv("REQUIREMENT_NLI_DEVICE") or "cpu"
        if not re.fullmatch(r"cpu|cuda(?::\d+)?|mps", self.device):
            raise ValueError("REQUIREMENT_NLI_DEVICE must be cpu, cuda, cuda:N, or mps.")
        if type(max_length) is not int or not 16 <= max_length <= 512:
            raise ValueError("NLI max_length must be an integer between 16 and 512.")
        if type(local_files_only) is not bool:
            raise ValueError("local_files_only must be a boolean.")
        if (tokenizer is None) != (model is None):
            raise ValueError("Inject both the NLI tokenizer and model, or neither.")
        self.local_files_only = local_files_only
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model
        self.id2label: dict[int, str] = {}
        self._torch = None
        self._ready = False

    @property
    def fingerprint(self) -> dict:
        return {
            "adapter": "requirement_nli_v2", "model": self.model_name,
            "revision": self.revision, "device": self.device,
            "local_files_only": self.local_files_only, "max_pair_tokens": self.max_length,
            "baseline": "pretrained general NLI; no custom job-matching calibration",
        }

    @staticmethod
    def _labels(config) -> dict[int, str]:
        try:
            raw = config.id2label
            if not isinstance(raw, dict) or len(raw) != 3:
                raise ValueError
            labels = {int(index): str(label).strip().casefold() for index, label in raw.items()}
            expected = {"entailment", "neutral", "contradiction"}
            if set(labels) != {0, 1, 2} or set(labels.values()) != expected:
                raise ValueError
            inverse = getattr(config, "label2id", None)
            if inverse:
                normalized = {str(label).strip().casefold(): int(index) for label, index in inverse.items()}
                if normalized != {label: index for index, label in labels.items()}:
                    raise ValueError
            if getattr(config, "num_labels", 3) != 3:
                raise ValueError
            return labels
        except (ValueError, TypeError, AttributeError):
            raise ModelProviderError(
                "The cached NLI model has ambiguous or invalid label mappings. "
                "Use weights explicitly labelled entailment, neutral and contradiction."
            ) from None

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        try:
            import torch
            if self.tokenizer is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                options = {"local_files_only": self.local_files_only, "trust_remote_code": False}
                if self.revision:
                    options["revision"] = self.revision
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, **options)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **options)
            self.id2label = self._labels(self.model.config)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._torch = torch
            self._ready = True
        except ModelProviderError:
            raise
        except Exception:
            # Loader messages can contain private paths or remote response bodies.
            raise ModelProviderError(
                "The NLI classifier could not load its local dependencies/weights. "
                "Install requirements-classifier.txt and provide a complete cached "
                "REQUIREMENT_NLI_MODEL/revision; automatic model downloads are disabled "
                "by the classifier application mode. CPU is the default device."
            ) from None

    def classify(self, context: str, hypothesis: str) -> dict:
        if not isinstance(context, str) or not context.strip() or not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("NLI context and hypothesis must be non-empty text.")
        self._ensure_loaded()
        try:
            inputs = self.tokenizer(
                context, hypothesis, return_tensors="pt", truncation=False, add_special_tokens=True,
            )
            length = int(inputs["input_ids"].shape[-1])
            limits = [self.max_length]
            for value in (getattr(self.tokenizer, "model_max_length", None), getattr(self.model.config, "max_position_embeddings", None)):
                if type(value) is int and 0 < value < 1_000_000:
                    limits.append(value)
            if length > min(limits):
                raise NLIInputTooLong(
                    f"The NLI sentence pair contains {length} tokens, exceeding the "
                    f"{min(limits)}-token allowance; no input was truncated."
                )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                outputs = self.model(**inputs)
            if tuple(outputs.logits.shape) != (1, 3):
                raise ModelProviderError("The cached NLI model returned an invalid prediction shape.")
            probabilities = self._torch.softmax(outputs.logits, dim=-1)[0]
            results = {self.id2label[index]: float(value.item()) for index, value in enumerate(probabilities)}
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in results.values()):
                raise ModelProviderError("The NLI model returned invalid probabilities.")
            predicted = max(results, key=results.get)
            return {"label": predicted, "confidence": results[predicted], "probabilities": results}
        except ModelProviderError:
            raise
        except Exception:
            raise ModelProviderError("Local NLI inference failed; check the cached tokenizer/model and device configuration.") from None
