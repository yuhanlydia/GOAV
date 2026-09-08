"""Trainer-exact global token-mean policy loss."""

from __future__ import annotations

import numpy as np


def model_logits(model_output):
    """Extract logits from Transformers-style objects, mappings or tuples."""
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]
    if isinstance(model_output, (tuple, list)) and model_output:
        return model_output[0]
    raise ValueError("causal LM output does not contain logits")


def response_token_mask(attention_mask, prompt_lengths):
    """Mask shifted causal targets belonging to each response."""
    if attention_mask.ndim != 2 or prompt_lengths.ndim != 1 or len(prompt_lengths) != len(attention_mask):
        raise ValueError("attention mask and prompt lengths do not align")
    if hasattr(attention_mask, "new_tensor"):
        import torch
        attended_rank = attention_mask.to(dtype=torch.long).cumsum(dim=1)[:, 1:]
        return attention_mask[:, 1:].to(dtype=torch.float32) * (attended_rank > prompt_lengths[:, None]).to(dtype=torch.float32)
    attention = np.asarray(attention_mask)
    lengths = np.asarray(prompt_lengths)
    attended_rank = np.cumsum(attention.astype(np.int64), axis=1)[:, 1:]
    return attention[:, 1:].astype(float) * (attended_rank > lengths[:, None])


def causal_lm_selected_log_probs(model_output, input_ids):
    logits = model_logits(model_output)
    if logits.shape[:-1] != input_ids.shape:
        raise ValueError("causal LM logits and input IDs do not align")
    shifted_logits = logits[:, :-1]
    targets = input_ids[:, 1:]
    if hasattr(shifted_logits, "log_softmax"):
        return shifted_logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    shifted = np.asarray(shifted_logits, dtype=float)
    maximum = shifted.max(axis=-1, keepdims=True)
    log_probs = shifted - maximum - np.log(np.exp(shifted - maximum).sum(axis=-1, keepdims=True))
    return np.take_along_axis(log_probs, np.asarray(targets)[..., None], axis=-1).squeeze(-1)


def token_mean_loss(selected_log_probabilities, rewards, token_mask):
    """Return ``-sum(logp * reward * mask) / sum(mask)`` across the whole batch.

    The denominator deliberately is not a mean of per-candidate means. This is
    the registered trainer estimand and must also be used to construct ``L``.
    """
    if selected_log_probabilities.shape != token_mask.shape:
        raise ValueError("log-probabilities and trainer token mask must align")
    if rewards.ndim != 1 or rewards.shape[0] != selected_log_probabilities.shape[0]:
        raise ValueError("one scalar reward is required per candidate")
    denominator = token_mask.sum()
    scalar_denominator = float(denominator.detach().cpu()) if hasattr(denominator, "detach") else float(denominator)
    if scalar_denominator <= 0:
        raise ValueError("trainer token mask must contain an active token")
    expanded_rewards = rewards.unsqueeze(-1) if hasattr(rewards, "unsqueeze") else rewards[..., None]
    return -(selected_log_probabilities * expanded_rewards * token_mask).sum() / denominator


def causal_lm_token_mean_loss(model_output, input_ids, rewards, response_mask):
    return token_mean_loss(causal_lm_selected_log_probs(model_output, input_ids), rewards, response_mask)
