import numpy as np
from types import SimpleNamespace

from goav.estimators import aipw_labels, clean_gradient, ht_labels
from goav.subsets import inclusion_probabilities, subset_matrix


def test_ht_and_aipw_literal_labels():
    audited = np.array([1, 0, 1])
    pi = np.array([0.5, 0.25, 1.0])
    labels = np.array([1.0, 0.0, 0.0])
    imputed = np.array([0.4, 0.3, 0.2])
    np.testing.assert_allclose(ht_labels(labels, audited, pi), [2.0, 0.0, 0.0])
    np.testing.assert_allclose(aipw_labels(imputed, audited, pi, labels), [1.6, 0.3, 0.0])


def test_aipw_is_exactly_design_unbiased_and_gradient_aligned():
    subsets = subset_matrix(3)
    probabilities = np.array([0.03, 0.07, 0.11, 0.09, 0.18, 0.12, 0.17, 0.23])
    pi, _ = inclusion_probabilities(subsets, probabilities)
    labels = np.array([1.0, 0.0, 1.0])
    imputed = np.array([0.2, 0.6, 0.7])
    expected = sum(probability * aipw_labels(imputed, audited, pi, labels) for probability, audited in zip(probabilities, subsets, strict=True))
    np.testing.assert_allclose(expected, labels, atol=1e-12)
    influence = np.array([[1.0, -2.0, 0.5], [0.2, 0.4, -0.3]])
    np.testing.assert_allclose(sum(probability * clean_gradient(influence, aipw_labels(imputed, audited, pi, labels)) for probability, audited in zip(probabilities, subsets, strict=True)), influence @ labels)


def test_estimators_fail_closed_on_zero_propensity():
    import pytest
    with pytest.raises(ValueError, match="positive"):
        aipw_labels(np.zeros(2), np.ones(2), np.array([1.0, 0.0]), np.ones(2))


def test_observed_only_estimators_are_finite_for_empty_audit():
    empty = SimpleNamespace(audited=np.zeros(3, dtype=int), revealed_indices=np.array([], dtype=int), revealed_labels=np.array([], dtype=float))
    pi = np.array([0.2, 0.3, 0.4])
    np.testing.assert_array_equal(ht_labels(empty, pi), np.zeros(3))
    np.testing.assert_allclose(aipw_labels(np.array([0.1, 0.4, 0.8]), empty, pi), [0.1, 0.4, 0.8])


def test_observed_only_estimators_scatter_revealed_values_and_reject_nonfinite():
    observed = SimpleNamespace(audited=np.array([0, 1, 0]), revealed_indices=np.array([1]), revealed_labels=np.array([1.0]))
    pi = np.array([0.2, 0.5, 0.4])
    np.testing.assert_allclose(ht_labels(observed, pi), [0.0, 2.0, 0.0])
    np.testing.assert_allclose(aipw_labels(np.array([0.1, 0.4, 0.8]), observed, pi), [0.1, 1.6, 0.8])
    bad = SimpleNamespace(audited=np.array([1]), revealed_indices=np.array([0]), revealed_labels=np.array([np.nan]))
    import pytest
    with pytest.raises(ValueError, match="finite"):
        ht_labels(bad, np.array([0.5]))
