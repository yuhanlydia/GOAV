import numpy as np
import pytest

from goav.audit import run_design_audit
from goav.bank import CandidateGroup, CheapTest, TrustedSidecar
from goav.baselines import CONTROLLED_DESIGNS, build_design
from goav.design import design_risk, solve_design
from goav.events import AuditEventLog
from goav.estimators import aipw_labels, ht_labels
from goav.subsets import inclusion_probabilities, subset_matrix


def test_k8_enumerates_all_256_subsets_including_empty():
    subsets = subset_matrix(8)
    assert subsets.shape == (256, 8)
    assert subsets[0].tolist() == [0] * 8
    assert subsets[-1].tolist() == [1] * 8
    assert len({tuple(row) for row in subsets}) == 256


def test_exact_first_and_second_order_propensities():
    subsets = subset_matrix(2)
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    pi_i, pi_ij = inclusion_probabilities(subsets, probabilities)
    np.testing.assert_allclose(pi_i, [0.6, 0.7])
    np.testing.assert_allclose(pi_ij, [[0.6, 0.4], [0.4, 0.7]])


def test_joint_solver_has_full_support_exact_budget_and_floor():
    covariance = np.full((8, 8), 0.08)
    np.fill_diagonal(covariance, 0.25)
    influence = np.arange(1, 25, dtype=float).reshape(3, 8) / 24
    design = solve_design(covariance, influence, expected_budget=0.8, inclusion_floor=0.05, exploration=0.1)
    assert np.all(design.probabilities > 0)
    assert np.isclose(design.probabilities.sum(), 1)
    assert np.isclose(design.pi_i.sum(), 0.8, atol=1e-8)
    assert np.all(design.pi_i >= 0.05 - 1e-10)
    np.testing.assert_allclose(np.diag(design.pi_ij), design.pi_i)
    assert design.probabilities[0] > 0
    assert design.optimization.feasible
    assert design.optimization.converged
    assert not design.optimization.global_optimum_certified
    with pytest.raises(ValueError, match="inclusion floor"):
        solve_design(covariance, influence, expected_budget=0.8, inclusion_floor=0.11)


def test_joint_design_and_risk_are_candidate_permutation_invariant():
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(5, 5))
    covariance = raw @ raw.T / 20
    influence = rng.normal(size=(4, 5))
    design = solve_design(covariance, influence, expected_budget=1.5, inclusion_floor=0.02, exploration=0.2)
    permutation = np.array([3, 0, 4, 1, 2])
    permuted = solve_design(covariance[np.ix_(permutation, permutation)], influence[:, permutation], expected_budget=1.5, inclusion_floor=0.02, exploration=0.2)
    assert np.isclose(design_risk(design, covariance, influence), design_risk(permuted, covariance[np.ix_(permutation, permutation)], influence[:, permutation]))
    np.testing.assert_allclose(permuted.pi_i, design.pi_i[permutation], atol=1e-8)


def test_all_controlled_designs_share_interface_and_top_k_is_biased():
    expected = {"uniform", "entropy", "coverage_kill", "factorized_neyman", "bayes_voi", "joint", "oracle", "full", "top_k"}
    assert set(CONTROLLED_DESIGNS) == expected
    covariance = np.eye(4) * 0.2
    influence = np.ones((2, 4))
    features = {"mean": np.array([0.1, 0.4, 0.8, 0.5]), "coverage_kill": np.arange(4.0)}
    for name in expected - {"full"}:
        design = build_design(name, covariance, influence, expected_budget=2, inclusion_floor=0.05, features=features)
        assert np.isclose(design.pi_i.sum(), 2)
        assert design.biased is (name == "top_k")
    full = build_design("full", covariance, influence, expected_budget=4, inclusion_floor=0.05, features=features)
    np.testing.assert_allclose(full.pi_i, 1)


def test_monte_carlo_draws_match_exact_propensities():
    design = solve_design(np.eye(4), np.ones((2, 4)), expected_budget=1.2, inclusion_floor=0.05)
    rng = np.random.default_rng(17)
    draws = design.subsets[rng.choice(len(design.subsets), size=100_000, p=design.probabilities)]
    np.testing.assert_allclose(draws.mean(axis=0), design.pi_i, atol=0.005)
    empirical_second = np.einsum("ni,nj->ij", draws, draws) / len(draws)
    np.testing.assert_allclose(empirical_second, design.pi_ij, atol=0.005)


def test_local_risk_optimizer_beats_reviewer_feasible_counterexample():
    covariance = np.full((8, 8), 0.2)
    np.fill_diagonal(covariance, 0.25)
    influence = np.ones((1, 8))
    design = solve_design(covariance, influence, expected_budget=0.8, inclusion_floor=0.02, exploration=1e-4)
    risk = design_risk(design, covariance, influence)
    assert risk <= 7.36
    assert risk <= min(design.optimization.baseline_risks.values()) + 1e-8
    assert design.name == "joint_local_risk_optimized"


def test_design_owns_immutable_arrays_and_recomputes_cached_propensities():
    design = solve_design(np.eye(3), np.ones((1, 3)), expected_budget=0.9, inclusion_floor=0.02)
    for array in (design.subsets, design.probabilities, design.pi_i, design.pi_ij):
        assert not array.flags.writeable
    with pytest.raises(ValueError, match="first-order propensities"):
        from goav.design import AuditDesign
        AuditDesign(design.name, design.subsets, design.probabilities, design.pi_i + 0.01, design.pi_ij, design.expected_budget, design.inclusion_floor, optimization=design.optimization)


def test_run_design_audit_commits_design_before_reading_outcome(tmp_path):
    tests = tuple(CheapTest(f"{kind}-{i}", kind, "x") for kind, count in (("curated", 12), ("generated", 12), ("fault_targeted", 12), ("metamorphic", 8), ("control", 4)) for i in range(count))
    group = CandidateGroup("p", "c", tuple(f"candidate-{i}" for i in range(8)), tests, np.zeros((8, 48), dtype=int))
    sidecar = TrustedSidecar.create({group.split_group: np.arange(8) % 2})
    design = solve_design(np.eye(8), np.ones((2, 8)), expected_budget=0.8, inclusion_floor=0.05)
    log = AuditEventLog(tmp_path / "events.jsonl")
    result = run_design_audit("run", group, design, sidecar, log, np.random.default_rng(3), rng_stream="seed:3", bank_hash="sha256:" + "a" * 64, draw_index=0)
    events = log.read()
    assert events[0].design_id == result.design_id == events[1].design_id
    assert type(events[0]).__name__ == "AuditDesignEvent"
    assert type(events[1]).__name__ == "AuditOutcomeEvent"
    assert all(np.isfinite(events[1].outcomes))
    assert result.revealed_indices.shape == result.revealed_labels.shape
    assert np.isfinite(result.revealed_labels).all()
    assert result.audited.sum() == 0
    assert np.isfinite(ht_labels(result, design.pi_i)).all()
    assert np.isfinite(aipw_labels(np.full(8, 0.5), result, design.pi_i)).all()


def test_audit_event_identity_separates_draws_and_revalidates_serialized_propensities(tmp_path):
    tests = tuple(CheapTest(f"{kind}-{i}", kind, "x") for kind, count in (("curated", 12), ("generated", 12), ("fault_targeted", 12), ("metamorphic", 8), ("control", 4)) for i in range(count))
    group = CandidateGroup("p", "checkpoint-a", tuple(f"c-{i}" for i in range(8)), tests, np.zeros((8, 48), dtype=int))
    sidecar = TrustedSidecar.create({group.split_group: np.ones(8)})
    design = solve_design(np.eye(8), np.ones((1, 8)), expected_budget=0.8, inclusion_floor=0.02)
    log = AuditEventLog(tmp_path / "identity.jsonl")
    first = run_design_audit("run", group, design, sidecar, log, np.random.default_rng(1), rng_stream="seed:1/draw:0", bank_hash="sha256:" + "d" * 64, draw_index=0)
    second = run_design_audit("run", group, design, sidecar, log, np.random.default_rng(2), rng_stream="seed:1/draw:1", bank_hash="sha256:" + "d" * 64, draw_index=1)
    assert first.design_id != second.design_id
    lines = (tmp_path / "identity.jsonl").read_text().splitlines()
    import json
    payload = json.loads(lines[0])
    payload["pi_i"][0] += 0.1
    lines[0] = json.dumps(payload)
    (tmp_path / "identity.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="propensit"):
        log.read()
