"""Stable exact binary pairwise Ising enumeration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class IsingMoments:
    log_partition: float
    mean: np.ndarray
    covariance: np.ndarray
    probabilities: np.ndarray
    states: np.ndarray


def enumerate_binary(k: int) -> np.ndarray:
    if not 0 <= k <= 16:
        raise ValueError("exact enumeration supports K at most 16")
    values = np.arange(1 << k, dtype=np.uint32)[:, None]
    return ((values >> np.arange(k, dtype=np.uint32)) & 1).astype(np.int64)


def ising_moments(unary: np.ndarray, pairwise: np.ndarray) -> IsingMoments:
    h = np.asarray(unary, dtype=float)
    j = np.asarray(pairwise, dtype=float)
    if h.ndim != 1 or h.size > 16:
        raise ValueError("unary must be a vector with K at most 16")
    if j.shape != (h.size, h.size):
        raise ValueError("pairwise must be K by K")
    if not np.allclose(j, j.T, atol=1e-12):
        raise ValueError("pairwise interactions must be symmetric")
    if not np.allclose(np.diag(j), 0.0, atol=1e-12):
        raise ValueError("pairwise diagonal must be zero")
    states = enumerate_binary(h.size).astype(float)
    log_weights = states @ h + 0.5 * np.einsum("si,ij,sj->s", states, j, states)
    log_partition = float(logsumexp(log_weights))
    probabilities = np.exp(log_weights - log_partition)
    mean = probabilities @ states
    second = np.einsum("s,si,sj->ij", probabilities, states, states)
    covariance = second - np.outer(mean, mean)
    return IsingMoments(log_partition, mean, covariance, probabilities, states.astype(np.int8))
