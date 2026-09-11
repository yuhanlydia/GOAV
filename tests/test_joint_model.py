import math
import sys

import numpy as np

from goav.crossfit import crossfit_predictions
from goav.ising import enumerate_binary, ising_moments
from goav.metrics import brier_score, covariance_rmse, expected_calibration_error, joint_nll
from goav.outcomes import _objective_and_gradient, fit_fold


def test_exact_ising_partition_mean_and_covariance_match_hand_enumeration():
    h = np.log([2.0, 3.0])
    j = np.array([[0.0, math.log(5.0)], [math.log(5.0), 0.0]])
    moments = ising_moments(h, j)
    assert enumerate_binary(2).tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert np.isclose(moments.log_partition, math.log(36.0))
    np.testing.assert_allclose(moments.mean, [32 / 36, 33 / 36])
    expected_e12 = 30 / 36
    expected_cov = np.array([
        [(32 / 36) * (4 / 36), expected_e12 - (32 / 36) * (33 / 36)],
        [expected_e12 - (32 / 36) * (33 / 36), (33 / 36) * (3 / 36)],
    ])
    np.testing.assert_allclose(moments.covariance, expected_cov)
    np.testing.assert_allclose(moments.probabilities, [1 / 36, 2 / 36, 3 / 36, 30 / 36])


def test_ising_rejects_large_or_asymmetric_models():
    import pytest
    with pytest.raises(ValueError, match="at most 16"):
        enumerate_binary(17)
    with pytest.raises(ValueError, match="symmetric"):
        ising_moments(np.zeros(2), np.array([[0.0, 1.0], [0.0, 0.0]]))


def test_candidate_permutation_equivariance_for_exact_and_fitted_model():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(10, 4, 3))
    labels = (features[..., 0] > 0).astype(float)
    model = fit_fold(features, labels, maxiter=30)
    permutation = np.array([2, 0, 3, 1])
    original = model.predict_moments(features[:1])[0]
    permuted = model.predict_moments(features[:1, permutation])[0]
    np.testing.assert_allclose(permuted.mean, original.mean[permutation], atol=1e-7)
    np.testing.assert_allclose(permuted.covariance, original.covariance[np.ix_(permutation, permutation)], atol=1e-7)


def test_ising_fit_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(3, 4, 2))
    labels = (rng.random((3, 4)) > 0.5).astype(float)
    parameters = rng.normal(size=features.shape[2] + 2)
    value, gradient = _objective_and_gradient(parameters, features, labels)
    epsilon = 1e-6
    numerical = np.empty_like(parameters)
    for index in range(len(parameters)):
        plus = parameters.copy(); plus[index] += epsilon
        minus = parameters.copy(); minus[index] -= epsilon
        numerical[index] = (_objective_and_gradient(plus, features, labels)[0] - _objective_and_gradient(minus, features, labels)[0]) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


def test_crossfit_never_shares_problem_checkpoint_groups():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(8, 3, 2))
    labels = (features[..., 0] > 0).astype(float)
    groups = [(f"p-{i // 2}", f"ckpt-{i % 2}") for i in range(8)]
    result = crossfit_predictions(features, labels, groups, n_folds=4, maxiter=10)
    assert result.means.shape == (8, 3)
    assert result.covariances.shape == (8, 3, 3)
    for fold in result.folds:
        assert set(fold.train_groups).isdisjoint(fold.test_groups)
        assert set(fold.train_base_ids).isdisjoint(fold.test_base_ids)
        assert fold.test_indices


def test_crossfit_keeps_all_checkpoints_and_duplicate_cluster_in_one_fold():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(8, 2, 2))
    labels = (features[..., 0] > 0).astype(float)
    groups = [
        ("p-a", "ckpt-1", "cluster-a"), ("p-a", "ckpt-2", "cluster-a"),
        ("p-alias", "ckpt-1", "cluster-a"), ("p-b", "ckpt-1", "cluster-b"),
        ("p-c", "ckpt-1", "cluster-c"), ("p-c", "ckpt-2", "cluster-c"),
        ("p-d", "ckpt-1", "cluster-d"), ("p-e", "ckpt-1", "cluster-e"),
    ]
    result = crossfit_predictions(features, labels, groups, n_folds=3, maxiter=5)
    for fold in result.folds:
        assert set(fold.train_base_ids).isdisjoint(fold.test_base_ids)
    cluster_folds = {next(fold.fold for fold in result.folds if index in fold.test_indices) for index in (0, 1, 2)}
    assert len(cluster_folds) == 1


def test_joint_metrics_are_literal_and_cpu_core_is_torch_free():
    states = enumerate_binary(2)
    probs = np.array([[0.1, 0.2, 0.3, 0.4]])
    labels = np.array([[1, 1]])
    means = np.array([[0.6, 0.7]])
    assert np.isclose(joint_nll(labels, probs, states), -math.log(0.4))
    assert np.isclose(brier_score(labels, means), (0.4**2 + 0.3**2) / 2)
    assert expected_calibration_error(np.array([0.1, 0.9]), np.array([0, 1]), n_bins=2) == 0.1
    assert covariance_rmse(np.eye(2), np.zeros((2, 2))) == math.sqrt(0.5)
    assert "torch" not in sys.modules
