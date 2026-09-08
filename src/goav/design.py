"""Full-subset audit designs with exact propensity recomputation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .subsets import inclusion_probabilities, subset_matrix


@dataclass(frozen=True)
class OptimizationDiagnostics:
    method: str
    converged: bool
    feasible: bool
    iterations: int
    message: str
    objective: float
    baseline_risks: Mapping[str, float]
    global_optimum_certified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_risks", MappingProxyType(dict(self.baseline_risks)))


@dataclass(frozen=True)
class AuditDesign:
    name: str
    subsets: np.ndarray
    probabilities: np.ndarray
    pi_i: np.ndarray
    pi_ij: np.ndarray
    expected_budget: float
    inclusion_floor: float
    biased: bool = False
    optimization: OptimizationDiagnostics | None = None

    def __post_init__(self) -> None:
        for field_name, dtype in (("subsets", np.int64), ("probabilities", float), ("pi_i", float), ("pi_ij", float)):
            value = np.array(getattr(self, field_name), dtype=dtype, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, field_name, value)
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        q = self.probabilities
        if q.ndim != 1 or len(q) == 0 or len(q) & (len(q) - 1):
            raise ValueError("subset probability vector length must be a power of two")
        k = int(np.log2(len(q)))
        canonical = subset_matrix(k)
        if self.subsets.shape != canonical.shape or not np.array_equal(self.subsets, canonical):
            raise ValueError("subsets must be the complete canonical enumeration")
        if not all(np.isfinite(value).all() for value in (q, self.pi_i, self.pi_ij)):
            raise ValueError("design arrays must be finite")
        if np.any(q < 0) or not np.isclose(q.sum(), 1.0, atol=1e-10):
            raise ValueError("subset probabilities must be non-negative and sum to one")
        recomputed_i, recomputed_ij = inclusion_probabilities(canonical, q)
        if self.pi_i.shape != (k,) or not np.allclose(self.pi_i, recomputed_i, atol=1e-10, rtol=1e-10):
            raise ValueError("cached first-order propensities do not match distribution")
        if self.pi_ij.shape != (k, k) or not np.allclose(self.pi_ij, recomputed_ij, atol=1e-10, rtol=1e-10):
            raise ValueError("cached second-order propensities do not match distribution")
        if not np.isclose(recomputed_i.sum(), self.expected_budget, atol=1e-8):
            raise ValueError("design expected budget mismatch")
        if np.any(recomputed_i < self.inclusion_floor - 1e-10):
            raise ValueError("design inclusion floor violation")

    def validate(self, *, require_full_support: bool = True) -> None:
        self._validate_consistency()
        if require_full_support and np.any(self.probabilities <= 0):
            raise ValueError("theoretical design lacks full subset support")
        if self.optimization is not None and not self.optimization.feasible:
            raise ValueError("risk optimizer returned an infeasible design")


def independent_distribution(inclusion: np.ndarray, subsets: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(inclusion, dtype=float)
    s = subset_matrix(len(p)) if subsets is None else np.asarray(subsets, dtype=float)
    log_q = s @ np.log(np.clip(p, 1e-300, 1.0)) + (1 - s) @ np.log(np.clip(1 - p, 1e-300, 1.0))
    return np.exp(log_q - logsumexp(log_q))


def calibrated_inclusions(scores: np.ndarray, expected_budget: float, inclusion_floor: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    k = len(values)
    if not k * inclusion_floor <= expected_budget < k:
        raise ValueError("inclusion floor is infeasible for expected budget")
    scale = max(float(np.std(values)), 1.0)
    normalized = (values - np.mean(values)) / scale
    low, high = -80.0, 80.0
    for _ in range(160):
        offset = (low + high) / 2
        probabilities = inclusion_floor + (1 - inclusion_floor) * expit(normalized + offset)
        if probabilities.sum() < expected_budget:
            low = offset
        else:
            high = offset
    return inclusion_floor + (1 - inclusion_floor) * expit(normalized + (low + high) / 2)


def _risk_from_q(q: np.ndarray, subsets: np.ndarray, weighted_covariance: np.ndarray, *, gradient: bool = False):
    pi_i = q @ subsets
    pi_ij = np.einsum("s,si,sj->ij", q, subsets, subsets)
    denominator = np.outer(pi_i, pi_i)
    value = float(np.sum(weighted_covariance * (pi_ij / denominator - 1.0)))
    if not gradient:
        return value
    first = np.einsum("ij,si,sj->s", weighted_covariance / denominator, subsets, subsets)
    weighted_second = weighted_covariance * pi_ij
    left = np.sum(weighted_second / (pi_i[:, None] ** 2 * pi_i[None, :]), axis=1)
    right = np.sum(weighted_second / (pi_i[:, None] * pi_i[None, :] ** 2), axis=0)
    return value, first - subsets @ (left + right)


def _cardinality_design(subsets: np.ndarray, expected_budget: float) -> np.ndarray:
    low, high = int(np.floor(expected_budget)), int(np.ceil(expected_budget))
    q = np.zeros(len(subsets), dtype=float)
    if low == high:
        selected = subsets.sum(axis=1) == low
        q[selected] = 1 / selected.sum()
    else:
        low_rows = subsets.sum(axis=1) == low
        high_rows = subsets.sum(axis=1) == high
        high_weight = expected_budget - low
        q[low_rows] = (1 - high_weight) / low_rows.sum()
        q[high_rows] = high_weight / high_rows.sum()
    return q


def _tilt_to_budget(q: np.ndarray, cardinalities: np.ndarray, expected_budget: float) -> np.ndarray:
    positive = np.clip(np.asarray(q, dtype=float), 0.0, None)
    if positive.sum() <= 0:
        raise ValueError("optimizer returned an empty distribution")
    low, high = -80.0, 80.0
    with np.errstate(divide="ignore"):
        log_q = np.where(positive > 0, np.log(positive), -np.inf)
    for _ in range(160):
        middle = (low + high) / 2
        logits = log_q + middle * cardinalities
        tilted = np.exp(logits - logsumexp(logits))
        if tilted @ cardinalities < expected_budget:
            low = middle
        else:
            high = middle
    logits = log_q + ((low + high) / 2) * cardinalities
    return np.exp(logits - logsumexp(logits))


def solve_design(covariance: np.ndarray, influence: np.ndarray, *, expected_budget: float, inclusion_floor: float, exploration: float = 0.05, temperature: float = 1.0, name: str = "joint_local_risk_optimized") -> AuditDesign:
    """Find a feasible local minimum of declared risk over all subset weights."""
    sigma = np.asarray(covariance, dtype=float)
    geometry = np.asarray(influence, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1] or geometry.ndim != 2 or geometry.shape[1] != sigma.shape[0]:
        raise ValueError("covariance/influence shapes do not align")
    if not np.isfinite(sigma).all() or not np.isfinite(geometry).all() or not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance and influence must be finite and covariance symmetric")
    k = sigma.shape[0]
    if not 0 < expected_budget < k:
        raise ValueError("full-support design expected budget must be in (0, K)")
    if inclusion_floor <= 0 or inclusion_floor * k > expected_budget + 1e-12:
        raise ValueError("inclusion floor is infeasible for expected budget")
    if not 0 <= exploration < 1 or temperature != 1.0:
        raise ValueError("exploration must be in [0,1) and temperature must equal one")
    subsets = subset_matrix(k).astype(float)
    cardinalities = subsets.sum(axis=1)
    weighted_covariance = sigma * (geometry.T @ geometry)
    symmetric = independent_distribution(np.full(k, expected_budget / k), subsets)
    cardinality = _cardinality_design(subsets, expected_budget)
    starts = {"independent_uniform": symmetric, "adjacent_cardinality": cardinality}
    mix = max(float(exploration), 1e-8)
    feasible_baselines = {key: _tilt_to_budget((1 - mix) * value + mix * symmetric, cardinalities, expected_budget) for key, value in starts.items()}
    baseline_risks = {key: _risk_from_q(value, subsets, weighted_covariance) for key, value in feasible_baselines.items()}
    constraints = (
        {"type": "eq", "fun": lambda q: q.sum() - 1.0, "jac": lambda q: np.ones_like(q)},
        {"type": "eq", "fun": lambda q: q @ cardinalities - expected_budget, "jac": lambda q: cardinalities},
        {"type": "ineq", "fun": lambda q: q @ subsets - inclusion_floor, "jac": lambda q: subsets.T},
    )
    candidates: list[tuple[np.ndarray, object]] = []
    for start in starts.values():
        result = minimize(lambda q: _risk_from_q(q, subsets, weighted_covariance, gradient=True), start, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * len(start), constraints=constraints, options={"maxiter": 500, "ftol": 1e-7, "disp": False})
        candidates.append((_tilt_to_budget(result.x, cardinalities, expected_budget), result))
    candidates.extend((value, None) for value in starts.values())
    feasible_candidates = []
    for candidate, result in candidates:
        probabilities = _tilt_to_budget((1 - mix) * candidate + mix * symmetric, cardinalities, expected_budget)
        pi_i, _ = inclusion_probabilities(subsets, probabilities)
        if pi_i.min() >= inclusion_floor - 1e-9:
            feasible_candidates.append((probabilities, result, _risk_from_q(probabilities, subsets, weighted_covariance)))
    if not feasible_candidates:
        raise RuntimeError("risk optimizer found no feasible full-support design")
    probabilities, selected_result, objective = min(feasible_candidates, key=lambda item: item[2])
    pi_i, pi_ij = inclusion_probabilities(subsets, probabilities)
    successful = [result for _, result, _ in feasible_candidates if result is not None and result.success]
    diagnostics = OptimizationDiagnostics("SLSQP-full-subset-local", bool(successful), True, max((int(result.nit) for result in successful), default=0), str(selected_result.message) if selected_result is not None else "best feasible declared baseline", float(objective), baseline_risks, False)
    design = AuditDesign(name, subsets.astype(np.int64), probabilities, pi_i, pi_ij, float(expected_budget), float(inclusion_floor), optimization=diagnostics)
    design.validate()
    return design


def design_risk(design: AuditDesign, covariance: np.ndarray, influence: np.ndarray) -> float:
    design.validate(require_full_support=not design.biased and design.name != "full")
    sigma = np.asarray(covariance, dtype=float)
    geometry = np.asarray(influence, dtype=float)
    if sigma.shape != design.pi_ij.shape or geometry.ndim != 2 or geometry.shape[1] != len(design.pi_i):
        raise ValueError("design/covariance/influence shapes do not align")
    return _risk_from_q(design.probabilities, design.subsets.astype(float), sigma * (geometry.T @ geometry))


def design_from_probabilities(name: str, probabilities: np.ndarray, *, expected_budget: float, inclusion_floor: float, biased: bool = False) -> AuditDesign:
    q = np.asarray(probabilities, dtype=float)
    if len(q) == 0 or len(q) & (len(q) - 1):
        raise ValueError("subset probability vector length must be a power of two")
    subsets = subset_matrix(int(np.log2(len(q))))
    pi_i, pi_ij = inclusion_probabilities(subsets, q)
    return AuditDesign(name, subsets, q, pi_i, pi_ij, float(expected_budget), float(inclusion_floor), biased)
