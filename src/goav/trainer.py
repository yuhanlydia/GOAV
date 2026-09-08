"""Optional PEFT/QLoRA one-update training integration."""

from __future__ import annotations

from dataclasses import dataclass

from .loss import causal_lm_token_mean_loss, token_mean_loss


@dataclass(frozen=True)
class OnlineTrainingBatch:
    input_ids: object
    labels: object
    response_mask: object


@dataclass(frozen=True)
class TrainingEvidence:
    loss: float
    active_tokens: int
    parameter_changed: bool


def attach_lora(model, *, rank: int = 32, alpha: int = 64, target_modules=None):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("PEFT is required for QLoRA training; install goav[model]") from exc
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM, target_modules=target_modules)
    return get_peft_model(model, config)


def qlora_one_update(model, optimizer, selected_log_probabilities, rewards, token_mask) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = token_mean_loss(selected_log_probabilities, rewards, token_mask)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def causal_qlora_one_update(model, optimizer, batch: OnlineTrainingBatch) -> TrainingEvidence:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for an executed online update") from exc
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("online update requires trainable parameters")
    if not torch.isfinite(batch.labels).all() or not torch.isfinite(batch.response_mask).all():
        raise ValueError("online labels and response mask must be finite")
    active_tokens = int(batch.response_mask.sum().detach().cpu())
    if active_tokens <= 0:
        raise ValueError("online update requires active response tokens")
    before = [parameter.detach().clone() for parameter in parameters]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = causal_lm_token_mean_loss(model(batch.input_ids), batch.input_ids, batch.labels, batch.response_mask)
    if not torch.isfinite(loss):
        raise ValueError("online loss must be finite")
    loss.backward()
    optimizer.step()
    changed = any(not torch.equal(previous, current.detach()) for previous, current in zip(before, parameters, strict=True))
    if not changed:
        raise RuntimeError("optimizer step did not change any trainable parameter")
    return TrainingEvidence(float(loss.detach().cpu()), active_tokens, True)
