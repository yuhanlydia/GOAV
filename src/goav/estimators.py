"""HT/AIPW labels and clean score-gradient composition."""

from __future__ import annotations

import numpy as np


def _inputs(audited: np.ndarray, propensities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(audited, dtype=float)
    pi = np.asarray(propensities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if d.shape != pi.shape or y.shape != pi.shape:
        raise ValueError("estimator inputs must have matching shapes")
    if np.any(pi <= 0):
        raise ValueError("inclusion propensities must be positive")
    if not np.isfinite(pi).all() or not np.isfinite(y).all() or not np.isin(d, [0.0, 1.0]).all():
        raise ValueError("dense estimator inputs must be finite with a binary audit mask")
    return d, pi, y


def _observed_inputs(observation, propensities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audited = np.asarray(observation.audited, dtype=float)
    indices = np.asarray(observation.revealed_indices, dtype=np.int64)
    values = np.asarray(observation.revealed_labels, dtype=float)
    pi = np.asarray(propensities, dtype=float)
    if audited.shape != pi.shape or not np.isin(audited, [0.0, 1.0]).all():
        raise ValueError("observation mask and propensities must align")
    if not np.array_equal(indices, np.flatnonzero(audited)) or values.shape != indices.shape:
        raise ValueError("revealed values must exactly match audited indices")
    if np.any(pi <= 0) or not np.isfinite(pi).all() or not np.isfinite(values).all():
        raise ValueError("propensities and revealed labels must be finite with positive propensities")
    return indices, values, pi


def ht_labels(labels_or_observation, audited_or_propensities: np.ndarray, propensities: np.ndarray | None = None) -> np.ndarray:
    if propensities is not None:
        d, pi, y = _inputs(audited_or_propensities, propensities, labels_or_observation)
        result = np.zeros_like(y, dtype=float)
        selected = np.flatnonzero(d)
        result[selected] = y[selected] / pi[selected]
        return result
    indices, values, pi = _observed_inputs(labels_or_observation, audited_or_propensities)
    result = np.zeros_like(pi)
    result[indices] = values / pi[indices]
    return result


def aipw_labels(imputed: np.ndarray, audited_or_observation, propensities: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    mu = np.asarray(imputed, dtype=float)
    if labels is not None:
        d, pi, y = _inputs(audited_or_observation, propensities, labels)
        if mu.shape != y.shape or not np.isfinite(mu).all():
            raise ValueError("finite imputation must match labels")
        result = mu.copy()
        selected = np.flatnonzero(d)
        result[selected] += (y[selected] - mu[selected]) / pi[selected]
        return result
    indices, values, pi = _observed_inputs(audited_or_observation, propensities)
    if mu.shape != pi.shape or not np.isfinite(mu).all():
        raise ValueError("finite imputation must match propensities")
    result = mu.copy()
    result[indices] += (values - mu[indices]) / pi[indices]
    return result


def clean_gradient(influence: np.ndarray, labels: np.ndarray) -> np.ndarray:
    matrix = np.asarray(influence, dtype=float)
    y = np.asarray(labels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(y):
        raise ValueError("influence matrix and labels do not align")
    return matrix @ y
