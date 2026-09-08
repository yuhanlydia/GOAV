"""Lazy model backends for fake CPU and real Transformers execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class BackendAnalysis:
    imputed: np.ndarray
    covariance: np.ndarray
    influence: np.ndarray
    candidate_token_ids: np.ndarray
    response_mask: np.ndarray
    loss_denominator: int


class PolicyBackend(Protocol):
    def analyze(self, view: Any) -> BackendAnalysis: ...


class DeterministicPolicyBackend:
    """Deterministic backend used by download-free integration runs."""

    def __init__(self, feature_dim: int = 4):
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim

    def analyze_candidate_ids(self, candidate_ids: Sequence[str]) -> BackendAnalysis:
        columns = []
        for identifier in candidate_ids:
            digest = hashlib.sha256(identifier.encode("utf-8")).digest()
            columns.append([(digest[index] / 255.0) * 2 - 1 for index in range(self.feature_dim)])
        influence = np.asarray(columns, dtype=float).T
        logits = influence[0]
        imputed = 1 / (1 + np.exp(-logits))
        diagonal = imputed * (1 - imputed)
        shared = 0.15 * np.sqrt(np.outer(diagonal, diagonal))
        covariance = shared
        np.fill_diagonal(covariance, diagonal)
        token_ids = np.asarray([[hashlib.sha256(identifier.encode("utf-8")).digest()[offset] % 31 for offset in range(3)] for identifier in candidate_ids], dtype=np.int64)
        response_mask = np.asarray([[1.0, 1.0, float(index % 2 == 0)] for index in range(len(candidate_ids))])
        return BackendAnalysis(imputed, covariance, influence, token_ids, response_mask, int(response_mask.sum()))

    def analyze(self, view: Any) -> BackendAnalysis:
        return self.analyze_candidate_ids(view.candidate_ids)


@dataclass
class TransformersPolicyBackend:
    model_id: str
    revision: str
    tokenizer: Any
    model: Any

    def generate(self, prompts: Sequence[str], **generation_kwargs: Any) -> list[str]:
        encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.model.generate(**encoded, **generation_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def token_log_probabilities(self, input_ids, attention_mask=None):
        from .loss import causal_lm_selected_log_probs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return causal_lm_selected_log_probs(outputs, input_ids)

    def analyze(self, view: Any) -> BackendAnalysis:
        from .gradient import score_geometry
        from .loss import response_token_mask
        responses = getattr(view, "candidate_texts", None)
        prompt = getattr(view, "prompt", None)
        if not responses or prompt is None:
            raise ValueError("Transformers analysis requires prepared prompt and candidate_texts")
        full_text = [prompt + response for response in responses]
        encoded = self.tokenizer(full_text, return_tensors="pt", padding=True)
        prompt_encoded = self.tokenizer([prompt] * len(responses), return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1).to(device)
        mask = response_token_mask(attention, prompt_lengths)
        influence = score_geometry(self.model, input_ids, mask, attention)
        cheap = np.asarray(view.cheap_outcomes, dtype=float)
        imputed = np.clip(cheap.mean(axis=1), 1e-4, 1 - 1e-4)
        covariance = np.diag(imputed * (1 - imputed))
        return BackendAnalysis(imputed, covariance, influence, input_ids[:, 1:].detach().cpu().numpy(), mask.detach().cpu().numpy(), int(mask.sum().detach().cpu()))


def load_transformers_backend(model_id: str, revision: str, *, quantization: str | None = None, device_map: str | dict[str, Any] = "auto") -> TransformersPolicyBackend:
    if not revision:
        raise ValueError("a pinned model revision is required")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and torch model extras are required; install goav[model]") from exc
    kwargs: dict[str, Any] = {"revision": revision, "device_map": device_map, "trust_remote_code": False}
    if quantization:
        if quantization not in {"nf4", "int8"}:
            raise ValueError("quantization must be nf4 or int8")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("bitsandbytes-compatible Transformers is required for quantization") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "nf4", load_in_8bit=quantization == "int8",
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return TransformersPolicyBackend(model_id, revision, tokenizer, model)


def load_registered_model(name: str, revision: str, *, quantization: str | None = None, device_map: str | dict[str, Any] = "auto") -> TransformersPolicyBackend:
    from .registry import MODELS
    if name not in MODELS:
        raise ValueError(f"unknown registered model: {name}")
    return load_transformers_backend(MODELS[name].hf_id, revision, quantization=quantization, device_map=device_map)
