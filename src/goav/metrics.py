"""Outcome-model diagnostics."""

from __future__ import annotations

import numpy as np


def joint_nll(labels: np.ndarray, probabilities: np.ndarray, states: np.ndarray) -> float:
    y = np.asarray(labels)
    p = np.atleast_2d(np.asarray(probabilities, dtype=float))
    s = np.asarray(states)
    if len(p) == 1 and len(y) > 1:
        p = np.repeat(p, len(y), axis=0)
    losses = []
    for target, row in zip(y, p, strict=True):
        matches = np.flatnonzero(np.all(s == target, axis=1))
        if len(matches) != 1:
            raise ValueError("label is not represented exactly once in state enumeration")
        losses.append(-np.log(np.clip(row[matches[0]], 1e-300, 1.0)))
    return float(np.mean(losses))


def brier_score(labels: np.ndarray, means: np.ndarray) -> float:
    return float(np.mean((np.asarray(labels, dtype=float) - np.asarray(means, dtype=float)) ** 2))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if p.shape != y.shape or n_bins < 1:
        raise ValueError("calibration inputs are invalid")
    indices = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for index in range(n_bins):
        selected = indices == index
        if selected.any():
            total += selected.mean() * abs(float(p[selected].mean() - y[selected].mean()))
    return float(round(total, 15))


def covariance_rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    predicted_array = np.asarray(predicted, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    if predicted_array.shape != observed_array.shape:
        raise ValueError("covariance shapes do not align")
    return float(np.sqrt(np.mean((predicted_array - observed_array) ** 2)))
