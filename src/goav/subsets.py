"""All-subset enumeration and exact inclusion propensities."""

from __future__ import annotations

import numpy as np

from .ising import enumerate_binary


def subset_matrix(k: int) -> np.ndarray:
    return enumerate_binary(k)


def inclusion_probabilities(subsets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(subsets, dtype=float)
    q = np.asarray(probabilities, dtype=float)
    if s.ndim != 2 or q.shape != (len(s),):
        raise ValueError("subset distribution shapes do not align")
    if np.any(q < 0) or not np.isclose(q.sum(), 1.0, atol=1e-10):
        raise ValueError("subset probabilities must be non-negative and sum to one")
    pi_i = q @ s
    pi_ij = np.einsum("s,si,sj->ij", q, s, s)
    return pi_i, pi_ij
