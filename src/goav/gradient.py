"""Score-gradient influence geometry aligned to the trainer denominator."""

from __future__ import annotations

import numpy as np

from .loss import causal_lm_selected_log_probs


def influence_from_token_scores(token_score_gradients: np.ndarray, token_mask: np.ndarray) -> np.ndarray:
    gradients = np.asarray(token_score_gradients, dtype=float)
    mask = np.asarray(token_mask, dtype=float)
    if gradients.ndim != 3 or mask.shape != gradients.shape[:2]:
        raise ValueError("token score gradients and mask shapes do not align")
    denominator = float(mask.sum())
    if denominator <= 0:
        raise ValueError("trainer token mask must contain an active token")
    return -(gradients * mask[..., None]).sum(axis=1).T / denominator


def score_geometry(model, input_ids, response_mask, attention_mask=None) -> np.ndarray:
    """Compute exact candidate score columns with one global token denominator."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the optional torch dependency is required for score_geometry") from exc
    output = model(input_ids) if attention_mask is None else model(input_ids=input_ids, attention_mask=attention_mask)
    selected = causal_lm_selected_log_probs(output, input_ids)
    denominator = response_mask.sum()
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("trainer token mask must contain an active token")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    columns = []
    for index in range(selected.shape[0]):
        contribution = -(selected[index] * response_mask[index]).sum() / denominator
        gradients = torch.autograd.grad(contribution, parameters, retain_graph=True, allow_unused=False)
        columns.append(torch.cat([gradient.reshape(-1) for gradient in gradients]))
    return torch.stack(columns, dim=1).detach().cpu().numpy()
