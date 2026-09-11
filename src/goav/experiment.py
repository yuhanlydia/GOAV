"""Connected frozen-bank diagnostics and fail-closed online execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from .audit import run_design_audit
from .bank import CandidateGroup, CheapTest, TrustedSidecar, bank_hash, trainer_view
from .baselines import build_design
from .design import design_risk
from .estimators import aipw_labels, clean_gradient, ht_labels
from .events import AuditDesignEvent, AuditEventLog, AuditOutcomeEvent
from .ledger import CostLedger
from .models import DeterministicPolicyBackend, PolicyBackend
from .statistics import holm_adjust, paired_hierarchical_randomization_pvalue

ARM_TO_DESIGN = {
    "uniform_ht": "uniform", "uniform_aipw": "uniform", "entropy_aipw": "entropy",
    "coverage_kill_aipw": "coverage_kill", "factorized_neyman_aipw": "factorized_neyman",
    "bayes_voi_aipw": "bayes_voi", "goav_joint_aipw": "joint",
    "oracle_covariance_goav": "oracle", "full_audit": "full", "top_k": "top_k",
}
ELIGIBLE_UNBIASED_ADAPTIVE_CONTROLS = frozenset({"entropy_aipw", "coverage_kill_aipw", "factorized_neyman_aipw", "bayes_voi_aipw"})


@dataclass(frozen=True)
class FrozenArmResult:
    name: str
    bank_hash: str
    expected_budget: float
    realized_budget: float
    design_events: int
    outcome_events: int
    predicted_risk: float | None
    predicted_risk_status: str
    gradient_nmse: float
    standardized_bias: float
    mean_cosine: float
    ess_fraction: float
    cost: dict[str, int | float | str]


@dataclass(frozen=True)
class FrozenRunResult:
    bank_hash: str
    arms: tuple[FrozenArmResult, ...]
    evidence_status: str
    gates: dict[str, object]


@dataclass(frozen=True)
class OnlineRunResult:
    algorithm: str
    status: str
    updates: int
    epochs_per_batch: int
    clipping: bool
    executed: bool
    loss: float | None = None
    active_tokens: int | None = None
    parameter_changed: bool = False


def synthetic_correlated_panel(tasks: int, *, seed: int, false_negative: float = 0.1, false_positive: float = 0.2, common_error: float = 0.6) -> tuple[list[CandidateGroup], TrustedSidecar]:
    if tasks < 2 or not all(0 <= value <= 1 for value in (false_negative, false_positive, common_error)):
        raise ValueError("synthetic panel settings are invalid")
    rng = np.random.default_rng(seed)
    tests = tuple(CheapTest(f"{kind}-{index}", kind, f"synthetic:{kind}:{index}") for kind, count in (("curated", 12), ("generated", 12), ("fault_targeted", 12), ("metamorphic", 8), ("control", 4)) for index in range(count))
    groups, labels_by_group = [], {}
    for task in range(tasks):
        task_latent = rng.normal()
        candidate_latent = task_latent + rng.normal(scale=0.8, size=8)
        labels = (candidate_latent > 0).astype(np.int64)
        outcomes = np.empty((8, 48), dtype=np.int8)
        for test_index in range(48):
            shared_flip = rng.random() < common_error * 0.2
            individual_rate = np.where(labels == 1, false_negative, false_positive) * (1 - common_error)
            flips = np.logical_xor(shared_flip, rng.random(8) < individual_rate)
            outcomes[:, test_index] = np.logical_xor(labels, flips)
        group = CandidateGroup(
            f"synthetic-{task}", "deterministic-checkpoint", tuple(f"task-{task}-candidate-{index}" for index in range(8)),
            tests, outcomes, prompt=f"synthetic problem {task}\n", candidate_texts=tuple(f"candidate program {index}" for index in range(8)),
        )
        groups.append(group)
        labels_by_group[group.split_group] = labels
    return groups, TrustedSidecar.create(labels_by_group)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else float(left.shape == right.shape and np.allclose(left, right))


def _clustered_task_means(errors: np.ndarray, base_ids: Sequence[str] | None = None, seed_ids: Sequence[int] | None = None) -> dict[int, dict[str, np.ndarray]]:
    values = np.asarray(errors, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("clustered errors must have shape (rows, draws, parameters)")
    bases = list(base_ids) if base_ids is not None else [str(index) for index in range(values.shape[0])]
    seeds = list(seed_ids) if seed_ids is not None else [0] * values.shape[0]
    if len(bases) != values.shape[0] or len(seeds) != values.shape[0]:
        raise ValueError("cluster identities must align with rows")
    return {
        seed: {
            base: np.mean(np.concatenate([values[index] for index in range(len(values)) if seeds[index] == seed and bases[index] == base], axis=0), axis=0)
            for base in sorted({bases[index] for index in range(len(values)) if seeds[index] == seed})
        }
        for seed in sorted(set(seeds))
    }


def _clustered_standard_error(errors: np.ndarray, base_ids: Sequence[str] | None = None, seed_ids: Sequence[int] | None = None) -> np.ndarray:
    """SE of the MC mean after equal-weighting draws within each task cluster."""
    values = np.asarray(errors, dtype=float)
    clustered = _clustered_task_means(values, base_ids, seed_ids)
    seed_means = np.stack([np.mean(list(tasks.values()), axis=0) for tasks in clustered.values()])
    if len(seed_means) > 1:
        return seed_means.std(axis=0, ddof=1) / np.sqrt(len(seed_means))
    task_means = np.stack(list(next(iter(clustered.values())).values()))
    if len(task_means) > 1:
        return task_means.std(axis=0, ddof=1) / np.sqrt(len(task_means))
    if values.shape[1] > 1:
        return values.reshape(-1, values.shape[-1]).std(axis=0, ddof=1) / np.sqrt(values.shape[1])
    return np.zeros(values.shape[2], dtype=float)


def run_frozen_bank(
    groups: Sequence[CandidateGroup], sidecar: TrustedSidecar, backend: PolicyBackend, arms: Sequence[str], *,
    expected_budget: float | Mapping[str, float], inclusion_floor: float, seed: int, event_directory: str | Path,
    design_draws: int = 1, oracle_covariances: Mapping[tuple[str, str], np.ndarray] | None = None,
    statistics_replicates: int = 1_000,
) -> FrozenRunResult:
    if not groups or design_draws < 1 or statistics_replicates < 1:
        raise ValueError("frozen execution requires groups and positive design_draws")
    if isinstance(expected_budget, Mapping):
        budgets = {arm: float(expected_budget[arm]) for arm in arms}
        controlled = [value for arm, value in budgets.items() if arm != "full_audit"]
        if len(set(controlled)) > 1:
            raise ValueError("controlled arms require a matched expected budget")
    else:
        budgets = {arm: float(expected_budget) for arm in arms}
    unknown = sorted(set(arms) - ARM_TO_DESIGN.keys())
    if unknown:
        raise ValueError(f"unsupported frozen controlled arms: {', '.join(unknown)}")
    if "top_k" in arms and not float(budgets["top_k"]).is_integer():
        raise ValueError("an integer expected budget is required for top_k")
    if "oracle_covariance_goav" in arms and oracle_covariances is None:
        raise ValueError("oracle covariance input is required for oracle_covariance_goav")
    common_hash = bank_hash(groups)
    analyses = [backend.analyze(trainer_view(group)) for group in groups]
    fitted_means = np.stack([analysis.imputed for analysis in analyses])
    fitted_covariances = np.stack([analysis.covariance for analysis in analyses])
    directory = Path(event_directory)
    directory.mkdir(parents=True, exist_ok=True)
    # One preregistered stream is replayed through every arm's inverse CDF.
    shared_uniforms = np.random.default_rng(seed).random((len(groups), design_draws))
    execution_order = sorted(range(len(groups)), key=lambda index: (groups[index].base_split_id, index))
    ordered_base_ids = [groups[index].base_split_id for index in execution_order]
    # Freeze every training-side selection object before evaluator label access.
    plans: dict[str, list[tuple[object, float | None, str]]] = {}
    for arm in arms:
        arm_plans = []
        for group_index, (group, analysis) in enumerate(zip(groups, analyses, strict=True)):
            design_name = ARM_TO_DESIGN[arm]
            budget = float(len(group.candidate_ids)) if design_name == "full" else budgets[arm]
            covariance = oracle_covariances[group.split_group] if design_name == "oracle" else fitted_covariances[group_index]
            design = build_design(design_name, covariance, analysis.influence, expected_budget=budget, inclusion_floor=inclusion_floor, features={"mean": fitted_means[group_index], "coverage_kill": group.cheap_outcomes.sum(axis=1).astype(float)})
            if design.biased or np.any(design.pi_i <= 0):
                arm_plans.append((design, None, "unavailable_zero_propensity"))
            else:
                risk = float(design_risk(design, covariance, analysis.influence))
                arm_plans.append((design, risk, "available"))
        plans[arm] = arm_plans
    results = []
    squared_errors: dict[str, np.ndarray] = {}
    for arm in arms:
        log = AuditEventLog(directory / f"{arm}.jsonl")
        rng = np.random.default_rng(seed)
        error_energies, target_energies, cosines = [], [], []
        ess_values, audit_counts = [], []
        ledger = CostLedger()
        actual_expected_budgets = []
        cluster_total = len(set(ordered_base_ids))
        cluster_mean_sum = None
        cluster_mean_squared_norm_sum = 0.0
        completed_clusters = 0
        current_base_id = None
        current_cluster_error_sum = None
        current_cluster_error_count = 0
        track_raw_error_moments = cluster_total == 1
        raw_error_sum = None
        raw_error_squared_norm_sum = 0.0
        raw_error_count = 0
        for group_index in execution_order:
            group = groups[group_index]
            analysis = analyses[group_index]
            design, _, _ = plans[arm][group_index]
            target = None
            group_error_sum = None
            for draw_index in range(design_draws):
                started = time.process_time()
                observation = run_design_audit(
                    f"frozen:{arm}", group, design, sidecar, log, rng,
                    rng_stream=f"seed:{seed}/group:{group_index}/draw:{draw_index}", bank_hash=common_hash, draw_index=draw_index,
                    uniform_draw=float(shared_uniforms[group_index, draw_index]),
                )
                evaluator_cpu_seconds = time.process_time() - started
                if target is None:
                    labels = sidecar.labels(group.split_group, evaluator_token=sidecar.evaluator_token)
                    target = clean_gradient(analysis.influence, labels)
                if arm == "uniform_ht":
                    estimated_labels = ht_labels(observation, design.pi_i)
                elif arm == "top_k":
                    estimated_labels = fitted_means[group_index].copy()
                    estimated_labels[observation.revealed_indices] = observation.revealed_labels
                else:
                    estimated_labels = aipw_labels(fitted_means[group_index], observation, design.pi_i)
                estimate = clean_gradient(analysis.influence, estimated_labels)
                error = estimate - target
                error_energies.append(float(error @ error))
                target_energies.append(float(target @ target))
                cosines.append(_cosine(estimate, target))
                if group_error_sum is None:
                    group_error_sum = np.zeros_like(error)
                group_error_sum += error
                if track_raw_error_moments:
                    if raw_error_sum is None:
                        raw_error_sum = np.zeros_like(error)
                    raw_error_sum += error
                    raw_error_squared_norm_sum += float(error @ error)
                    raw_error_count += 1
                weights = np.zeros_like(design.pi_i)
                weights[observation.revealed_indices] = 1 / design.pi_i[observation.revealed_indices]
                audit_count = int(observation.audited.sum())
                if audit_count:
                    ess_values.append(float((weights.sum() ** 2) / (audit_count * (weights @ weights))))
                audit_counts.append(audit_count)
                ledger.charge(tokens=analysis.loss_denominator, test_executions=audit_count, cpu_seconds=evaluator_cpu_seconds)
            assert group_error_sum is not None
            base_id = group.base_split_id
            if current_base_id is not None and base_id != current_base_id:
                assert current_cluster_error_sum is not None and current_cluster_error_count > 0
                cluster_mean = current_cluster_error_sum / current_cluster_error_count
                if cluster_mean_sum is None:
                    cluster_mean_sum = np.zeros_like(cluster_mean)
                cluster_mean_sum += cluster_mean
                cluster_mean_squared_norm_sum += float(cluster_mean @ cluster_mean)
                completed_clusters += 1
                current_cluster_error_sum = None
                current_cluster_error_count = 0
            current_base_id = base_id
            if current_cluster_error_sum is None:
                current_cluster_error_sum = np.zeros_like(group_error_sum)
            current_cluster_error_sum += group_error_sum
            current_cluster_error_count += design_draws
            actual_expected_budgets.append(design.expected_budget)
        assert current_cluster_error_sum is not None and current_cluster_error_count > 0
        cluster_mean = current_cluster_error_sum / current_cluster_error_count
        if cluster_mean_sum is None:
            cluster_mean_sum = np.zeros_like(cluster_mean)
        cluster_mean_sum += cluster_mean
        cluster_mean_squared_norm_sum += float(cluster_mean @ cluster_mean)
        completed_clusters += 1
        assert completed_clusters == cluster_total
        squared_errors[arm] = np.asarray(error_energies)
        error_energy = np.asarray(error_energies).reshape(len(groups), design_draws, 1)
        target_energy_rows = np.asarray(target_energies).reshape(len(groups), design_draws, 1)
        target_energy = float(np.mean([value for tasks in _clustered_task_means(target_energy_rows, ordered_base_ids).values() for value in tasks.values()]))
        mse = float(np.mean([value for tasks in _clustered_task_means(error_energy, ordered_base_ids).values() for value in tasks.values()]))
        bias_vector = cluster_mean_sum / completed_clusters
        if completed_clusters > 1:
            centered_squared_norm = max(
                cluster_mean_squared_norm_sum - completed_clusters * float(bias_vector @ bias_vector), 0.0
            )
            bias_standard_error_norm = np.sqrt(
                centered_squared_norm / (completed_clusters - 1) / completed_clusters
            )
        elif design_draws > 1:
            assert raw_error_sum is not None and raw_error_count > 1
            raw_mean = raw_error_sum / raw_error_count
            raw_variance_norm = max(
                (raw_error_squared_norm_sum - raw_error_count * float(raw_mean @ raw_mean)) / (raw_error_count - 1),
                0.0,
            )
            bias_standard_error_norm = np.sqrt(raw_variance_norm / design_draws)
        else:
            bias_standard_error_norm = 0.0
        risk_values = [item[1] for item in plans[arm]]
        risk_statuses = {item[2] for item in plans[arm]}
        predicted_risk = float(np.mean(risk_values)) if all(value is not None for value in risk_values) else None
        predicted_risk_status = next(iter(risk_statuses)) if len(risk_statuses) == 1 else "partially_unavailable"
        design_events, outcome_events = log.counts()
        results.append(FrozenArmResult(
            arm, common_hash, float(np.mean(actual_expected_budgets)), float(np.mean(audit_counts)),
            design_events, outcome_events,
            predicted_risk, predicted_risk_status, mse / max(target_energy, 1e-12), float(np.linalg.norm(bias_vector) / max(bias_standard_error_norm, 1e-12)),
            float(np.mean(cosines)), float(np.mean(ess_values)) if ess_values else 0.0, {**ledger.as_dict(), "cpu_seconds_provenance": "measured_process_time", "ess_provenance": "kish_normalized_within_nonempty_audit"},
        ))
    by_name = {result.name: result for result in results}
    gates: dict[str, object] = {"support_violations": 0.0}
    if "goav_joint_aipw" in by_name:
        eligible = sorted(name for name in by_name if name in ELIGIBLE_UNBIASED_ADAPTIVE_CONTROLS)
        gates.update({"nmse_comparators": eligible, "nmse_comparator_count": len(eligible), "ess_gate": bool(by_name["goav_joint_aipw"].ess_fraction >= 0.3)})
        if eligible:
            baseline_nmse = min(by_name[name].gradient_nmse for name in eligible)
            reduction = 1 - by_name["goav_joint_aipw"].gradient_nmse / baseline_nmse if baseline_nmse > 0 else 0.0
            task_ids = [base_id for base_id in ordered_base_ids for _ in range(design_draws)]
            seed_ids = [seed] * len(task_ids)
            raw_p = [paired_hierarchical_randomization_pvalue(squared_errors[name], squared_errors["goav_joint_aipw"], task_ids, seed_ids, n_replicates=statistics_replicates, seed=seed + index) for index, name in enumerate(eligible)]
            adjusted = holm_adjust(raw_p)
            adjusted_by_name = {name: float(value) for name, value in zip(eligible, adjusted, strict=True)}
            strongest = min(eligible, key=lambda name: by_name[name].gradient_nmse)
            gates.update({"nmse_reduction": float(reduction), "holm_adjusted_p_values": adjusted_by_name, "holm_alpha": 0.05, "nmse_gate": bool(reduction >= 0.1 and adjusted_by_name[strongest] <= 0.05)})
    evidence_status = "connected_synthetic_smoke_not_phase_evidence" if isinstance(backend, DeterministicPolicyBackend) else "frozen_diagnostics_not_sealed_evidence"
    return FrozenRunResult(common_hash, tuple(results), evidence_status, gates)


def run_online(backend: PolicyBackend, *, algorithm: str, updates: int = 1, epochs_per_batch: int = 1, clipping: bool = False, claim_identified: bool = False, training_batch=None, optimizer=None) -> OnlineRunResult:
    if algorithm.lower() == "nc_grpo":
        raise ValueError("nc_grpo is rejected; choose noise_corrected_grpo or noise_contrastive_grpo")
    if updates < 1 or epochs_per_batch < 1:
        raise ValueError("updates and epochs_per_batch must be positive")
    theoretical_arm = algorithm in {"uniform_ht", "uniform_aipw", "goav_joint_aipw", "oracle_covariance_goav"}
    eligible = theoretical_arm and updates == 1 and epochs_per_batch == 1 and not clipping
    if (training_batch is None) != (optimizer is None):
        raise ValueError("training_batch and optimizer must be provided together")
    if training_batch is None:
        if claim_identified:
            raise ValueError("identified status requires verified execution of a one-update causal-LM batch")
        return OnlineRunResult(algorithm, "planned_not_evidence", updates, epochs_per_batch, clipping, False)
    if not eligible:
        raise ValueError("the connected online runner supports only one unclipped theoretical update")
    if claim_identified:
        raise ValueError("identified status requires sealed audit provenance; caller-supplied labels are not sufficient")
    from .trainer import causal_qlora_one_update
    model = getattr(backend, "model", backend)
    evidence = causal_qlora_one_update(model, optimizer, training_batch)
    status = "executed_unverified_label_update_not_evidence" if evidence.parameter_changed else "planned_not_evidence"
    return OnlineRunResult(algorithm, status, updates, epochs_per_batch, clipping, True, evidence.loss, evidence.active_tokens, evidence.parameter_changed)
