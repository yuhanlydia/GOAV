"""Strict, side-effect-free experiment configuration loading."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from .registry import BASELINES, BENCHMARKS, MODELS
from .schema import ExperimentConfig
from .manifest import file_digest

OPTIONAL_DEPENDENCIES = ("torch", "transformers", "peft", "bitsandbytes", "datasets")
PROFILES = {"smoke", "16gb", "24gb", "h200_formal"}
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-f]{7,40}\Z")
FORMAL_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _require_digest(value: Any, name: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a valid lowercase sha256 digest")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dependency_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in OPTIONAL_DEPENDENCIES}


def _verify_execution_artifacts(raw: Mapping[str, Any], registration_root: str | Path | None) -> None:
    if registration_root is None:
        raise ValueError("formal execution requires a registration root")
    root = Path(registration_root).resolve()
    artifacts = raw.get("execution_artifacts")
    if type(artifacts) is not dict or set(artifacts) != {"bank_manifest", "phase_gate", "evaluator_attestation"}:
        raise ValueError("formal execution requires bank, gate and evaluator registration artifacts")
    loaded = {}
    for name, record in artifacts.items():
        if type(record) is not dict or set(record) != {"path", "digest"} or type(record["path"]) is not str:
            raise ValueError(f"invalid {name} artifact registration")
        _require_digest(record["digest"], f"{name} artifact digest")
        relative = Path(record["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or root not in path.parents or not path.is_file():
            raise ValueError(f"registered {name} artifact is missing or unsafe")
        if file_digest(path) != record["digest"]:
            raise ValueError(f"registered {name} artifact digest mismatch")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    if artifacts["bank_manifest"]["digest"] != raw["bank_manifest_hash"]:
        raise ValueError("bank manifest registration hash mismatch")
    gate = loaded["phase_gate"]
    if gate.get("passed") is not True or gate.get("gate_spec_hash") != raw["gate_spec_hash"]:
        raise ValueError("phase gate artifact is not a verified pass")
    attestation = loaded["evaluator_attestation"]
    required = ("process_isolation", "restricted_sandbox", "network_disabled")
    if attestation.get("status") != "verified" or any(attestation.get(key) is not True for key in required):
        raise ValueError("evaluator attestation does not verify required isolation")
    raise ValueError("external evaluator attestation signature verification is unavailable; formal execution is fail-closed")


def validate_experiment(raw: Mapping[str, Any], *, for_execution: bool = False, registration_root: str | Path | None = None) -> None:
    required = {"name", "profile", "formal", "model", "benchmark", "baselines", "seeds", "expected_audit_fraction"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"missing required configuration keys: {', '.join(missing)}")
    if type(raw["name"]) is not str or not raw["name"]:
        raise ValueError("name must be a non-empty string")
    if type(raw["profile"]) is not str:
        raise ValueError("profile must be a string")
    if type(raw["formal"]) is not bool:
        raise ValueError("formal must be a boolean")
    if type(raw["model"]) is not dict or type(raw["benchmark"]) is not dict:
        raise ValueError("model and benchmark must be mappings")
    if type(raw["baselines"]) is not list or not raw["baselines"] or not all(type(item) is str for item in raw["baselines"]):
        raise ValueError("baselines must be a non-empty list of strings")
    if type(raw["seeds"]) is not list or not raw["seeds"] or not all(type(seed) is int for seed in raw["seeds"]):
        raise ValueError("seeds must be a non-empty list of exact integers")
    if not _is_number(raw["expected_audit_fraction"]):
        raise ValueError("expected_audit_fraction must be a finite number")
    profile = raw["profile"]
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if raw["formal"] and profile != "h200_formal":
        raise ValueError("formal experiments require the h200_formal profile")
    if raw["formal"] and len(set(raw["seeds"])) < 3:
        raise ValueError("formal experiments require at least three independent seeds")
    model = raw["model"]
    benchmark = raw["benchmark"]
    if model.get("name") not in MODELS:
        raise ValueError(f"unknown model: {model.get('name')}")
    revision_pattern = FORMAL_REVISION_RE if raw["formal"] else REVISION_RE
    if type(model.get("revision")) is not str or revision_pattern.fullmatch(model["revision"]) is None:
        raise ValueError("model revision must be a resolved 40-hex commit" if raw["formal"] else "model revision must be a resolved 7-40-hex commit")
    if benchmark.get("name") not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {benchmark.get('name')}")
    if type(benchmark.get("revision")) is not str or revision_pattern.fullmatch(benchmark["revision"]) is None:
        raise ValueError("benchmark revision must be a resolved 40-hex commit" if raw["formal"] else "benchmark revision must be a resolved 7-40-hex commit")
    _require_digest(benchmark.get("digest"), "benchmark digest")
    arms = [str(arm).lower() for arm in raw["baselines"]]
    if "nc_grpo" in arms:
        raise ValueError("nc_grpo is ambiguous; use noise_corrected_grpo or noise_contrastive_grpo")
    unknown = sorted(set(arms) - BASELINES.keys())
    if unknown:
        raise ValueError(f"unknown baselines: {', '.join(unknown)}")
    fraction = float(raw["expected_audit_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError("expected_audit_fraction must be in (0, 1]")
    design = raw.get("design", {})
    if type(design) is not dict:
        raise ValueError("design must be a mapping")
    if "theoretical" in design and type(design["theoretical"]) is not bool:
        raise ValueError("design.theoretical must be a boolean")
    if design.get("theoretical", False) and design.get("full_support") is not True:
        raise ValueError("theoretical audit designs require full support")
    arm_budgets = raw.get("arm_expected_audit_fraction")
    if arm_budgets is not None:
        if set(arm_budgets) != set(raw["baselines"]):
            raise ValueError("arm budget keys must match configured baselines")
        if len({float(value) for value in arm_budgets.values()}) != 1 or not math.isclose(next(iter(map(float, arm_budgets.values()))), fraction, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("controlled arms require a matched expected budget")
    if raw["formal"]:
        activation = raw.get("formal_activation", "registered")
        if activation not in {"pending", "registered"}:
            raise ValueError("formal_activation must be pending or registered")
        pending = activation == "pending"
        if pending and for_execution:
            raise ValueError("pending formal registration cannot be used for execution")
        unavailable = [name for name in raw["baselines"] if not BASELINES[name].executable]
        if for_execution and unavailable:
            raise ValueError(f"formal execution has unresolved comparator provenance: {', '.join(unavailable)}")
        if type(raw.get("bootstrap_replicates")) is not int or raw["bootstrap_replicates"] < 10_000:
            raise ValueError("formal experiments require at least 10000 bootstrap replicates")
        if pending:
            if raw.get("bank_manifest_hash") != "unresolved" or raw.get("registered_config_hash") != "unresolved":
                raise ValueError("pending formal templates must use explicit unresolved artifact bindings")
        else:
            _require_digest(raw.get("bank_manifest_hash"), "bank manifest hash")
        gates = raw.get("gates")
        if type(gates) is not dict or not gates:
            raise ValueError("formal experiments require concrete gates")
        gate_hash = _require_digest(raw.get("gate_spec_hash"), "gate specification hash")
        if gate_hash != canonical_hash(gates):
            raise ValueError("gate specification hash does not match gates")
        abort = raw.get("abort")
        if type(abort) is not dict:
            raise ValueError("formal experiments require abort thresholds")
        limits = {"peak_memory_fraction": 0.9, "infrastructure_failure_fraction": 0.01, "cost_overrun_fraction": 0.25}
        for key, ceiling in limits.items():
            value = abort.get(key)
            if not _is_number(value) or not 0 < float(value) <= ceiling:
                raise ValueError(f"abort.{key} must be positive and no greater than {ceiling}")
        if raw.get("stage") == "learned_tests":
            if raw.get("fixed_pool_gate_passed") is not True:
                raise ValueError("learned-tests stage requires a passed fixed-pool gate")
            _require_digest(raw.get("fixed_pool_gate_artifact_hash"), "fixed-pool gate artifact hash")
        if not pending:
            registered_hash = _require_digest(raw.get("registered_config_hash"), "registered config hash")
            registered_payload = {key: value for key, value in raw.items() if key != "registered_config_hash"}
            if registered_hash != canonical_hash(registered_payload):
                raise ValueError("registered config hash does not match canonical configuration")
            if for_execution:
                _verify_execution_artifacts(raw, registration_root)


def load_experiment(path: str | Path, *, for_execution: bool = False, registration_root: str | Path | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment YAML must contain a mapping")
    validate_experiment(raw, for_execution=for_execution, registration_root=registration_root)
    return ExperimentConfig(
        name=str(raw["name"]), profile=str(raw["profile"]), formal=bool(raw["formal"]),
        model=dict(raw["model"]), benchmark=dict(raw["benchmark"]),
        baselines=tuple(str(x).lower() for x in raw["baselines"]),
        seeds=tuple(int(x) for x in raw["seeds"]),
        expected_audit_fraction=float(raw["expected_audit_fraction"]),
        config_hash=canonical_hash(raw), optional_dependencies=dependency_status(), raw=raw,
    )
