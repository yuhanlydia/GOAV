"""Permutation-equivariant exact joint outcome model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .ising import IsingMoments, ising_moments


@dataclass(frozen=True)
class ExactIsingOutcomeModel:
    intercept: float
    weights: np.ndarray
    interaction: float

    def _parameters(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(candidates, dtype=float)
        unary = self.intercept + x @ self.weights
        k = x.shape[0]
        pairwise = np.full((k, k), self.interaction, dtype=float)
        np.fill_diagonal(pairwise, 0.0)
        return unary, pairwise

    def predict_moments(self, features: np.ndarray) -> list[IsingMoments]:
        x = np.asarray(features, dtype=float)
        if x.ndim != 3:
            raise ValueError("features must have shape tasks by candidates by features")
        return [ising_moments(*self._parameters(row)) for row in x]


def fit_fold(features: np.ndarray, labels: np.ndarray, *, maxiter: int = 100) -> ExactIsingOutcomeModel:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.ndim != 3 or y.shape != x.shape[:2]:
        raise ValueError("features/labels shapes do not align")
    if x.shape[1] > 16 or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("exact fitting requires binary labels and K at most 16")

    def objective(parameters: np.ndarray) -> float:
        model = ExactIsingOutcomeModel(float(parameters[0]), parameters[1:-1], float(parameters[-1]))
        total = 0.0
        for row, target in zip(x, y, strict=True):
            unary, pairwise = model._parameters(row)
            moments = ising_moments(unary, pairwise)
            energy = target @ unary + 0.5 * target @ pairwise @ target
            total += moments.log_partition - float(energy)
        return total / len(x) + 1e-5 * float(parameters @ parameters)

    initial = np.zeros(x.shape[2] + 2, dtype=float)
    fitted = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": int(maxiter), "ftol": 1e-9})
    if not np.isfinite(fitted.fun):
        raise RuntimeError("joint outcome fit produced a non-finite objective")
    return ExactIsingOutcomeModel(float(fitted.x[0]), fitted.x[1:-1].copy(), float(fitted.x[-1]))


def build_torch_joint_model(feature_dim: int):
    """Construct an optional learned equivariant model without importing torch at module load."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the optional torch dependency is required for the learned joint model") from exc

    class TorchJointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unary = torch.nn.Linear(feature_dim, 1)
            self.interaction = torch.nn.Parameter(torch.zeros(()))

        def forward(self, features):
            unary = self.unary(features).squeeze(-1)
            k = features.shape[-2]
            eye = torch.eye(k, device=features.device, dtype=features.dtype)
            pairwise = self.interaction * (1 - eye)
            return unary, pairwise.expand(*features.shape[:-2], k, k)

    return TorchJointModel()
