import json
import tracemalloc

import numpy as np
import pytest

from goav.bank import CandidateGroup, CheapTest, TrustedSidecar
from goav.artifacts import ArtifactStore
from goav.benchmarks import JsonlBenchmark, load_evalplus
from goav.experiment import _clustered_standard_error, run_frozen_bank, run_online, synthetic_correlated_panel
from goav.models import DeterministicPolicyBackend
from goav.statistics import hierarchical_bootstrap, holm_adjust, paired_hierarchical_bootstrap, paired_hierarchical_randomization_pvalue


def _groups():
    tests = tuple(CheapTest(f"{kind}-{i}", kind, "test") for kind, count in (("curated", 12), ("generated", 12), ("fault_targeted", 12), ("metamorphic", 8), ("control", 4)) for i in range(count))
    return [CandidateGroup(f"p-{problem}", "checkpoint", tuple(f"p{problem}-c{i}" for i in range(8)), tests, np.zeros((8, 48), dtype=int)) for problem in range(2)]


class InspectingBackend(DeterministicPolicyBackend):
    def __init__(self):
        super().__init__(feature_dim=2)
        self.views = []

    def analyze(self, view):
        assert not hasattr(view, "trusted_labels")
        self.views.append(view.candidate_ids)
        return self.analyze_candidate_ids(view.candidate_ids)


def test_frozen_runner_uses_hidden_labels_common_banks_and_matched_expected_budgets(tmp_path):
    groups = _groups()
    sidecar = TrustedSidecar.create({group.split_group: np.arange(8) % 2 for group in groups})
    backend = InspectingBackend()
    result = run_frozen_bank(groups, sidecar, backend, ["uniform_aipw", "goav_joint_aipw"], expected_budget=0.8, inclusion_floor=0.05, seed=3, event_directory=tmp_path, design_draws=3)
    assert len(backend.views) == len(groups)
    assert {arm.bank_hash for arm in result.arms} == {result.bank_hash}
    assert {arm.expected_budget for arm in result.arms} == {0.8}
    assert all(arm.design_events == 3 * len(groups) and arm.outcome_events == 3 * len(groups) for arm in result.arms)
    for arm in result.arms:
        assert np.isfinite([arm.gradient_nmse, arm.standardized_bias, arm.mean_cosine, arm.ess_fraction, arm.realized_budget, arm.predicted_risk]).all()
        assert arm.cost["test_executions"] >= 0 and arm.cost["tokens"] > 0
        assert arm.cost["cpu_seconds_provenance"] == "measured_process_time"
    assert result.evidence_status == "connected_synthetic_smoke_not_phase_evidence"
    assert len({event.design_id for event in __import__("goav.events", fromlist=["AuditEventLog"]).AuditEventLog(tmp_path / "uniform_aipw.jsonl").read() if type(event).__name__ == "AuditDesignEvent"}) == 3 * len(groups)
    event_log = __import__("goav.events", fromlist=["AuditEventLog"]).AuditEventLog
    uniform_events = [event for event in event_log(tmp_path / "uniform_aipw.jsonl").read() if type(event).__name__ == "AuditDesignEvent"]
    goav_events = [event for event in event_log(tmp_path / "goav_joint_aipw.jsonl").read() if type(event).__name__ == "AuditDesignEvent"]
    assert [(event.rng_stream, event.uniform_draw) for event in uniform_events] == [(event.rng_stream, event.uniform_draw) for event in goav_events]
    assert result.gates["nmse_comparator_count"] == 0


def test_frozen_runner_streams_high_dimensional_draw_metrics(tmp_path):
    class WideBackend(DeterministicPolicyBackend):
        def analyze(self, view):
            analysis = super().analyze(view)
            return __import__("dataclasses").replace(
                analysis, influence=np.tile(analysis.influence, (5_000, 1))
            )

    groups = _groups()
    sidecar = TrustedSidecar.create({group.split_group: np.arange(8) % 2 for group in groups})
    tracemalloc.start()
    run_frozen_bank(
        groups, sidecar, WideBackend(feature_dim=4),
        ["uniform_aipw"], expected_budget=0.8, inclusion_floor=0.05,
        seed=3, event_directory=tmp_path, design_draws=60,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 30_000_000


def test_frozen_runner_streams_high_dimensional_cluster_bias_stats(tmp_path):
    class WideBackend(DeterministicPolicyBackend):
        shared_influence = None

        def analyze(self, view):
            analysis = super().analyze(view)
            if self.shared_influence is None:
                self.shared_influence = np.tile(analysis.influence, (10_000, 1))
            return __import__("dataclasses").replace(
                analysis, influence=self.shared_influence
            )

    template = _groups()[0]
    groups = [
        CandidateGroup(
            f"cluster-{index}", "checkpoint", tuple(f"c{index}-{candidate}" for candidate in range(8)),
            template.tests, template.cheap_outcomes,
        )
        for index in range(48)
    ]
    sidecar = TrustedSidecar.create({group.split_group: np.arange(8) % 2 for group in groups})
    tracemalloc.start()
    run_frozen_bank(
        groups, sidecar, WideBackend(feature_dim=4),
        ["uniform_aipw"], expected_budget=0.8, inclusion_floor=0.05,
        seed=3, event_directory=tmp_path, design_draws=2,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 15_000_000


def test_streamed_cluster_bias_matches_prechange_reference(tmp_path):
    groups = _groups()
    sidecar = TrustedSidecar.create({group.split_group: np.arange(8) % 2 for group in groups})
    result = run_frozen_bank(
        groups, sidecar, InspectingBackend(), ["uniform_aipw"],
        expected_budget=0.8, inclusion_floor=0.05, seed=9,
        event_directory=tmp_path, design_draws=3,
    )
    arm = result.arms[0]
    assert arm.standardized_bias == pytest.approx(0.8866277071256814)
    assert arm.gradient_nmse == pytest.approx(18.2228736219711)
    assert arm.mean_cosine == pytest.approx(0.5911561893722879)


def test_gate_uses_only_unbiased_adaptive_controls_and_bias_uses_cluster_mc_se(tmp_path):
    groups = _groups()
    sidecar = TrustedSidecar.create({group.split_group: np.arange(8) % 2 for group in groups})
    result = run_frozen_bank(groups, sidecar, InspectingBackend(), ["full_audit", "uniform_aipw", "entropy_aipw", "goav_joint_aipw"], expected_budget=0.8, inclusion_floor=0.05, seed=9, event_directory=tmp_path, design_draws=3)
    assert result.gates["nmse_comparators"] == ["entropy_aipw"]
    assert set(result.gates["holm_adjusted_p_values"]) == {"entropy_aipw"}
    errors = np.array([[[1.0], [3.0]], [[5.0], [7.0]]])
    # Task means are 2 and 6; the standard error of their equally weighted mean is 2.
    np.testing.assert_allclose(_clustered_standard_error(errors), [2.0])
    duplicated = np.array([[[1.0]], [[1.0]], [[1.0]], [[3.0]], [[3.0]], [[3.0]]])
    np.testing.assert_allclose(_clustered_standard_error(duplicated, ["a", "a", "a", "b", "b", "b"]), [1.0])


def test_frozen_runner_rejects_mismatched_arm_budgets(tmp_path):
    groups = _groups()
    sidecar = TrustedSidecar.create({group.split_group: np.ones(8) for group in groups})
    with pytest.raises(ValueError, match="matched expected budget"):
        run_frozen_bank(groups, sidecar, InspectingBackend(), ["uniform_aipw", "goav_joint_aipw"], expected_budget={"uniform_aipw": 0.8, "goav_joint_aipw": 1.6}, inclusion_floor=0.05, seed=1, event_directory=tmp_path)


def test_frozen_budget_truth_and_special_inputs_fail_closed(tmp_path):
    groups = _groups()
    sidecar = TrustedSidecar.create({group.split_group: np.ones(8) for group in groups})
    with pytest.raises(ValueError, match="integer.*top_k"):
        run_frozen_bank(groups, sidecar, InspectingBackend(), ["top_k"], expected_budget=0.8, inclusion_floor=0.05, seed=1, event_directory=tmp_path / "top")
    with pytest.raises(ValueError, match="oracle covariance"):
        run_frozen_bank(groups, sidecar, InspectingBackend(), ["oracle_covariance_goav"], expected_budget=0.8, inclusion_floor=0.05, seed=1, event_directory=tmp_path / "oracle")
    full = run_frozen_bank(groups, sidecar, InspectingBackend(), ["full_audit"], expected_budget=0.8, inclusion_floor=0.05, seed=1, event_directory=tmp_path / "full")
    assert full.arms[0].expected_budget == 8 and full.arms[0].realized_budget == 8
    top = run_frozen_bank(groups, sidecar, InspectingBackend(), ["top_k"], expected_budget=1, inclusion_floor=0.05, seed=1, event_directory=tmp_path / "top-valid")
    assert top.arms[0].predicted_risk is None
    assert top.arms[0].predicted_risk_status == "unavailable_zero_propensity"
    json.dumps(__import__("dataclasses").asdict(top), allow_nan=False)


def test_correlated_panel_is_reproducible_and_not_phase_evidence():
    groups_a, sidecar_a = synthetic_correlated_panel(64, seed=8)
    groups_b, sidecar_b = synthetic_correlated_panel(64, seed=8)
    np.testing.assert_array_equal(groups_a[0].cheap_outcomes, groups_b[0].cheap_outcomes)
    stacked = np.concatenate([group.cheap_outcomes for group in groups_a], axis=1)
    correlation = np.corrcoef(stacked)
    assert np.nanmean(correlation[np.triu_indices(8, 1)]) > 0.05
    assert sidecar_a.labels(groups_a[0].split_group, evaluator_token=sidecar_a.evaluator_token).shape == (8,)


def test_online_metadata_separates_identified_one_update_from_practical_approximations():
    backend = DeterministicPolicyBackend(feature_dim=2)
    exact = run_online(backend, algorithm="goav_joint_aipw", updates=1, epochs_per_batch=1, clipping=False)
    assert exact.status == "planned_not_evidence"
    assert not exact.executed
    practical = run_online(backend, algorithm="grpo", updates=2, epochs_per_batch=2, clipping=True)
    assert practical.status == "planned_not_evidence"
    with pytest.raises(ValueError, match="verified.*one-update"):
        run_online(backend, algorithm="goav_joint_aipw", updates=2, epochs_per_batch=1, clipping=False, claim_identified=True)
    with pytest.raises(ValueError, match="nc_grpo"):
        run_online(backend, algorithm="nc_grpo")


def test_caller_supplied_online_labels_can_never_claim_identified(monkeypatch):
    from goav.trainer import TrainingEvidence
    mutations = []
    def mutating_update(model, optimizer, batch):
        mutations.append("optimizer-step")
        return TrainingEvidence(1.25, 7, True)
    monkeypatch.setattr("goav.trainer.causal_qlora_one_update", mutating_update)
    backend = DeterministicPolicyBackend()
    result = run_online(backend, algorithm="goav_joint_aipw", training_batch=object(), optimizer=object())
    assert result.executed and result.parameter_changed
    assert result.status == "executed_unverified_label_update_not_evidence"
    assert mutations == ["optimizer-step"]
    with pytest.raises(ValueError, match="sealed audit provenance"):
        run_online(backend, algorithm="goav_joint_aipw", training_batch=object(), optimizer=object(), claim_identified=True)
    assert mutations == ["optimizer-step"]


def test_all_designs_are_frozen_before_any_trusted_label_access(tmp_path, monkeypatch):
    groups = _groups()
    sequence = []
    from goav import experiment as experiment_module
    original_build = experiment_module.build_design
    monkeypatch.setattr(experiment_module, "build_design", lambda *args, **kwargs: (sequence.append("design"), original_build(*args, **kwargs))[1])
    class OrderingSidecar(TrustedSidecar):
        def labels(self, *args, **kwargs):
            sequence.append("label")
            return super().labels(*args, **kwargs)
    base = TrustedSidecar.create({group.split_group: np.arange(8) % 2 for group in groups})
    sidecar = OrderingSidecar(base._labels, base.evaluator_token)
    run_frozen_bank(groups, sidecar, InspectingBackend(), ["uniform_aipw", "entropy_aipw"], expected_budget=0.8, inclusion_floor=0.05, seed=4, event_directory=tmp_path)
    assert sequence[:4] == ["design"] * 4
    assert "design" not in sequence[sequence.index("label"):]


def test_jsonl_benchmark_and_hierarchical_bootstrap_are_download_free(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('\n'.join(json.dumps({"task_id": f"t-{i}", "prompt": f"p-{i}"}) for i in range(4)) + '\n', encoding="utf-8")
    rows = list(JsonlBenchmark(path))
    assert [row.task_id for row in rows] == ["t-0", "t-1", "t-2", "t-3"]
    summary = hierarchical_bootstrap(np.array([1.0, 2.0, 3.0, 4.0]), ["a", "b", "c", "d"], [0, 0, 1, 1], n_replicates=500, seed=7)
    assert summary.estimate == 2.5
    assert summary.ci_low <= summary.estimate <= summary.ci_high


def test_named_benchmark_adapter_is_lazy_and_artifacts_are_immutable(tmp_path):
    import importlib.util
    if importlib.util.find_spec("datasets") is None:
        with pytest.raises(RuntimeError, match="datasets.*benchmark"):
            load_evalplus(revision="abc1234", split="test")
    store = ArtifactStore(tmp_path / "artifacts")
    path = store.write_json("run.json", {"status": "smoke"})
    with pytest.raises(FileExistsError, match="immutable"):
        store.write_json("run.json", {"status": "changed"})
    manifest = store.manifest()
    manifest.verify()
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tamper"):
        manifest.verify()


def test_artifact_store_prevents_escape_and_seal_overwrite_and_binds_hashes(tmp_path):
    from goav.artifacts import verify_checksum_index, write_checksum_index
    store = ArtifactStore(tmp_path / "run")
    with pytest.raises(ValueError, match="safe relative"):
        store.write_json("../escape.json", {})
    store.write_json("result.json", {"config_hash": "sha256:" + "a" * 64, "bank_hash": "sha256:" + "b" * 64})
    write_checksum_index(store.directory, config_hash="sha256:" + "a" * 64, bank_hash="sha256:" + "b" * 64)
    verify_checksum_index(store.directory, config_hash="sha256:" + "a" * 64, bank_hash="sha256:" + "b" * 64)
    with pytest.raises(FileExistsError, match="sealed"):
        write_checksum_index(store.directory, config_hash="sha256:" + "a" * 64, bank_hash="sha256:" + "b" * 64)
    with pytest.raises(PermissionError, match="sealed"):
        store.write_json("late.json", {})
    with pytest.raises(ValueError, match="bank hash"):
        verify_checksum_index(store.directory, config_hash="sha256:" + "a" * 64, bank_hash="sha256:" + "c" * 64)


def test_paired_statistics_equal_weight_tasks_then_seeds_and_holm_corrects():
    values = np.array([0.0, 10.0, 10.0, 100.0])
    tasks = ["a", "a", "b", "c"]
    seeds = [0, 0, 0, 1]
    summary = hierarchical_bootstrap(values, tasks, seeds, n_replicates=100, seed=2)
    assert summary.estimate == (7.5 + 100.0) / 2
    paired = paired_hierarchical_bootstrap(values + 2, values, tasks, seeds, n_replicates=100, seed=2)
    assert paired.estimate == 2.0
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert paired_hierarchical_randomization_pvalue(np.ones(20), np.zeros(20), list(range(20)), [0] * 20, n_replicates=1000, seed=4) < 0.01
