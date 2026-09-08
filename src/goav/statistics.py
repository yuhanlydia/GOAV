"""Task-paired hierarchical bootstrap summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapSummary:
    estimate: float
    ci_low: float
    ci_high: float
    standard_error: float
    replicates: int


def hierarchical_bootstrap(values: np.ndarray, task_ids: Sequence[Hashable], seed_ids: Sequence[Hashable], *, n_replicates: int = 10_000, seed: int = 0, confidence: float = 0.95) -> BootstrapSummary:
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or len(observations) != len(task_ids) or len(observations) != len(seed_ids):
        raise ValueError("bootstrap inputs must align")
    if n_replicates < 1 or not 0 < confidence < 1:
        raise ValueError("bootstrap settings are invalid")
    rng = np.random.default_rng(seed)
    unique_seeds = sorted(set(seed_ids), key=str)
    task_means: dict[Hashable, dict[Hashable, float]] = {}
    for seed_id in unique_seeds:
        tasks_for_seed = sorted({task_ids[index] for index in range(len(observations)) if seed_ids[index] == seed_id}, key=str)
        task_means[seed_id] = {
            task_id: float(np.mean([observations[index] for index in range(len(observations)) if seed_ids[index] == seed_id and task_ids[index] == task_id]))
            for task_id in tasks_for_seed
        }
    seed_means = [float(np.mean(list(task_means[seed_id].values()))) for seed_id in unique_seeds]
    draws = np.empty(n_replicates)
    for replicate in range(n_replicates):
        sampled_seed_indices = rng.integers(0, len(unique_seeds), size=len(unique_seeds))
        sampled_seed_means = []
        for sampled_seed_index in sampled_seed_indices:
            sampled_seed = unique_seeds[int(sampled_seed_index)]
            means = np.array(list(task_means[sampled_seed].values()), dtype=float)
            sampled_seed_means.append(float(np.mean(rng.choice(means, size=len(means), replace=True))))
        draws[replicate] = np.mean(sampled_seed_means)
    tail = (1 - confidence) / 2
    return BootstrapSummary(float(np.mean(seed_means)), float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail)), float(draws.std(ddof=1)) if n_replicates > 1 else 0.0, n_replicates)


def paired_hierarchical_bootstrap(left: np.ndarray, right: np.ndarray, task_ids: Sequence[Hashable], seed_ids: Sequence[Hashable], *, n_replicates: int = 10_000, seed: int = 0, confidence: float = 0.95) -> BootstrapSummary:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape:
        raise ValueError("paired observations must have identical shapes")
    return hierarchical_bootstrap(left_values - right_values, task_ids, seed_ids, n_replicates=n_replicates, seed=seed, confidence=confidence)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(np.minimum(1.0, values[order] * (len(values) - np.arange(len(values)))))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_hierarchical_randomization_pvalue(left: np.ndarray, right: np.ndarray, task_ids: Sequence[Hashable], seed_ids: Sequence[Hashable], *, n_replicates: int = 10_000, seed: int = 0) -> float:
    """One-sided paired test, sign-randomized at task clusters and equal-weighted by seed."""
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if differences.ndim != 1 or differences.shape != np.asarray(right).shape or len(differences) != len(task_ids) or len(differences) != len(seed_ids) or not np.isfinite(differences).all():
        raise ValueError("paired randomization inputs must be aligned finite vectors")
    if n_replicates < 1:
        raise ValueError("randomization replicates must be positive")
    unique_seeds = sorted(set(seed_ids), key=str)
    clustered: dict[Hashable, np.ndarray] = {}
    for seed_id in unique_seeds:
        tasks = sorted({task_ids[index] for index in range(len(differences)) if seed_ids[index] == seed_id}, key=str)
        clustered[seed_id] = np.asarray([
            np.mean([differences[index] for index in range(len(differences)) if seed_ids[index] == seed_id and task_ids[index] == task_id])
            for task_id in tasks
        ])
    observed = float(np.mean([values.mean() for values in clustered.values()]))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_replicates):
        null_seed_means = [float(np.mean(values * rng.choice((-1.0, 1.0), size=len(values)))) for values in clustered.values()]
        exceedances += float(np.mean(null_seed_means)) >= observed
    return float((exceedances + 1) / (n_replicates + 1))
