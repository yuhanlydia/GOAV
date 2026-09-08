"""Controlled audit designs through one exact-propensity interface."""

from __future__ import annotations

import numpy as np

from .design import AuditDesign, calibrated_inclusions, design_from_probabilities, independent_distribution, solve_design
from .subsets import inclusion_probabilities, subset_matrix

CONTROLLED_DESIGNS = {"uniform", "entropy", "coverage_kill", "factorized_neyman", "bayes_voi", "joint", "oracle", "full", "top_k"}


def _factorized(name: str, scores: np.ndarray, expected_budget: float, inclusion_floor: float) -> AuditDesign:
    p = calibrated_inclusions(scores, expected_budget, inclusion_floor)
    q = independent_distribution(p)
    design = design_from_probabilities(name, q, expected_budget=expected_budget, inclusion_floor=inclusion_floor)
    design.validate()
    return design


def build_design(name: str, covariance: np.ndarray, influence: np.ndarray, *, expected_budget: float, inclusion_floor: float, features: dict[str, np.ndarray] | None = None) -> AuditDesign:
    if name not in CONTROLLED_DESIGNS:
        raise ValueError(f"unknown controlled design: {name}")
    sigma = np.asarray(covariance, dtype=float)
    geometry = np.asarray(influence, dtype=float)
    k = sigma.shape[0]
    features = features or {}
    if name == "full":
        if not np.isclose(expected_budget, k):
            raise ValueError("full audit requires expected budget K")
        q = np.zeros(1 << k)
        q[-1] = 1.0
        return design_from_probabilities(name, q, expected_budget=k, inclusion_floor=inclusion_floor)
    if name == "top_k":
        count = int(round(expected_budget))
        if not np.isclose(count, expected_budget):
            raise ValueError("top_k requires an integer budget")
        scores = np.diag(geometry.T @ geometry) * np.diag(sigma)
        selected = np.argsort(scores, kind="stable")[-count:]
        subset = np.zeros(k, dtype=int)
        subset[selected] = 1
        q = np.zeros(1 << k)
        q[int(sum(int(bit) << index for index, bit in enumerate(subset)))] = 1.0
        return design_from_probabilities(name, q, expected_budget=expected_budget, inclusion_floor=0.0, biased=True)
    if name in {"joint", "oracle"}:
        optimized_name = "joint_local_risk_optimized" if name == "joint" else "oracle_local_risk_optimized"
        return solve_design(sigma, geometry, expected_budget=expected_budget, inclusion_floor=inclusion_floor, name=optimized_name)
    if name == "uniform":
        scores = np.zeros(k)
    elif name == "entropy":
        mean = np.asarray(features.get("mean", np.full(k, 0.5)), dtype=float)
        scores = -(mean * np.log(np.clip(mean, 1e-12, 1)) + (1 - mean) * np.log(np.clip(1 - mean, 1e-12, 1)))
    elif name == "coverage_kill":
        scores = np.asarray(features.get("coverage_kill", np.zeros(k)), dtype=float)
    elif name == "factorized_neyman":
        scores = np.sqrt(np.maximum(np.diag(geometry.T @ geometry) * np.diag(sigma), 0.0))
    else:
        scores = np.diag(sigma)
    return _factorized(name, scores, expected_budget, inclusion_floor)
