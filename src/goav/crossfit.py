"""Problem/checkpoint-grouped cross-fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .outcomes import fit_fold


@dataclass(frozen=True)
class CrossfitFold:
    fold: int
    train_groups: tuple[tuple[str, ...], ...]
    test_groups: tuple[tuple[str, ...], ...]
    train_base_ids: tuple[str, ...]
    test_base_ids: tuple[str, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class CrossfitPredictions:
    means: np.ndarray
    covariances: np.ndarray
    folds: tuple[CrossfitFold, ...]


def crossfit_predictions(features: np.ndarray, labels: np.ndarray, groups: Sequence[tuple[str, ...]], *, n_folds: int = 5, maxiter: int = 100) -> CrossfitPredictions:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(groups) != len(x) or n_folds < 2:
        raise ValueError("cross-fit groups and fold count are invalid")
    if any(len(group) not in (2, 3) for group in groups):
        raise ValueError("groups must be (problem, checkpoint[, duplicate_cluster])")
    base_ids = [str(group[2] if len(group) == 3 else group[0]) for group in groups]
    unique = sorted(set(base_ids))
    if len(unique) < n_folds:
        raise ValueError("cross-fitting requires at least one group per fold")
    assignment = {base_id: index % n_folds for index, base_id in enumerate(unique)}
    means = np.empty_like(y, dtype=float)
    covariances = np.empty((len(x), x.shape[1], x.shape[1]), dtype=float)
    records = []
    group_array = np.array([assignment[base_id] for base_id in base_ids])
    for fold in range(n_folds):
        test_indices = np.flatnonzero(group_array == fold)
        train_indices = np.flatnonzero(group_array != fold)
        model = fit_fold(x[train_indices], y[train_indices], maxiter=maxiter)
        predictions = model.predict_moments(x[test_indices])
        means[test_indices] = np.stack([item.mean for item in predictions])
        covariances[test_indices] = np.stack([item.covariance for item in predictions])
        train_groups = tuple(sorted({groups[index] for index in train_indices}))
        test_groups = tuple(sorted({groups[index] for index in test_indices}))
        train_base_ids = tuple(sorted({base_ids[index] for index in train_indices}))
        test_base_ids = tuple(sorted({base_ids[index] for index in test_indices}))
        if set(train_base_ids) & set(test_base_ids):
            raise RuntimeError("base-problem/duplicate-cluster leakage across cross-fit fold")
        records.append(CrossfitFold(fold, train_groups, test_groups, train_base_ids, test_base_ids, tuple(map(int, test_indices))))
    return CrossfitPredictions(means, covariances, tuple(records))
