import importlib.util
import json
import sys

import pytest
import yaml

from goav.config import canonical_hash, load_experiment, validate_experiment
from goav.registry import BASELINES, BENCHMARKS, MODELS


def _write(tmp_path, payload):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_registries_expose_preregistered_models_datasets_and_arms_with_provenance():
    assert set(MODELS) == {"plumbing", "primary", "architecture_replication", "cross_family", "utrl_control", "reasonflux_4b", "reasonflux_7b"}
    assert MODELS["primary"].hf_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert set(BENCHMARKS) == {"synthetic", "codecontests_o", "taco_codecontests", "evalplus", "livecodebench_v6", "bigcodebench_hard", "held_out"}
    required = {"cheap_only", "uniform_ht", "uniform_aipw", "entropy_aipw", "coverage_kill_aipw", "factorized_neyman_aipw", "bayes_voi_aipw", "noise_corrected_grpo", "noise_contrastive_grpo", "goav_joint_aipw", "oracle_covariance_goav", "full_audit", "top_k", "grpo", "dr_grpo", "vpo", "b4", "codet", "cure", "utrl_4b", "cosplay"}
    assert required <= set(BASELINES)
    for name in required:
        spec = BASELINES[name]
        assert spec.paper and spec.repository and spec.revision and spec.license and spec.mode
        assert spec.provenance_status in {"resolved-local-protocol", "unresolved-upstream-pin"}
        if not spec.executable:
            assert "unresolved" in spec.revision and "unverified" in spec.license


def test_load_experiment_is_canonical_and_does_not_import_optional_ml_dependencies(tmp_path, monkeypatch):
    payload = {
        "name": "cpu-smoke",
        "profile": "16gb",
        "formal": False,
        "model": {"name": "plumbing", "revision": "0123456789abcdef"},
        "benchmark": {"name": "synthetic", "revision": "fedcba9876543210", "digest": "sha256:" + "a" * 64},
        "baselines": ["uniform_aipw", "goav_joint_aipw"],
        "seeds": [7],
        "expected_audit_fraction": 0.1,
    }
    before = set(sys.modules)
    config = load_experiment(_write(tmp_path, payload))
    after = set(sys.modules)
    assert config.name == "cpu-smoke"
    assert config.config_hash == canonical_hash(payload)
    assert not ({"torch", "transformers", "datasets", "peft", "bitsandbytes"} & (after - before))
    assert config.optional_dependencies["transformers"] == (importlib.util.find_spec("transformers") is not None)


@pytest.mark.parametrize("bad_name", ["nc_grpo", "NC_GRPO"])
def test_rejects_ambiguous_nc_grpo_alias(tmp_path, bad_name):
    path = _write(tmp_path, {
        "name": "bad", "profile": "16gb", "formal": False,
        "model": {"name": "plumbing", "revision": "abc1234"},
        "benchmark": {"name": "synthetic", "revision": "abc1234", "digest": "sha256:" + "b" * 64},
        "baselines": [bad_name], "seeds": [1], "expected_audit_fraction": 0.1,
    })
    with pytest.raises(ValueError, match="noise_corrected_grpo.*noise_contrastive_grpo"):
        load_experiment(path)


def test_validation_rejects_missing_pins_and_formal_local_profiles(tmp_path):
    base = {
        "name": "formal", "profile": "h200_formal", "formal": True,
        "model": {"name": "primary", "revision": "abc1234"},
        "benchmark": {"name": "codecontests_o", "revision": "abc1234", "digest": "sha256:" + "c" * 64},
        "baselines": ["goav_joint_aipw"], "seeds": [1, 2, 3], "expected_audit_fraction": 0.1,
    }
    missing_revision = json.loads(json.dumps(base))
    del missing_revision["model"]["revision"]
    with pytest.raises(ValueError, match="model revision"):
        load_experiment(_write(tmp_path, missing_revision))
    local_formal = json.loads(json.dumps(base))
    local_formal["profile"] = "24gb"
    with pytest.raises(ValueError, match="formal.*h200_formal"):
        load_experiment(_write(tmp_path, local_formal))
    too_few_seeds = json.loads(json.dumps(base))
    too_few_seeds["seeds"] = [1, 2]
    with pytest.raises(ValueError, match="three.*seeds"):
        validate_experiment(too_few_seeds)


def test_validation_rejects_non_supporting_theory_and_mismatched_arm_budgets(tmp_path):
    base = {
        "name": "theory", "profile": "smoke", "formal": False,
        "model": {"name": "plumbing", "revision": "abc1234"},
        "benchmark": {"name": "synthetic", "revision": "abc1234", "digest": "sha256:" + "d" * 64},
        "baselines": ["uniform_aipw", "goav_joint_aipw"], "seeds": [1], "expected_audit_fraction": 0.1,
        "candidate_count": 8,
    }
    no_support = {**base, "design": {"theoretical": True, "full_support": False}}
    with pytest.raises(ValueError, match="full support"):
        load_experiment(_write(tmp_path, no_support))
    mismatched = {**base, "arm_expected_audit_fraction": {"uniform_aipw": 0.1, "goav_joint_aipw": 0.2}}
    with pytest.raises(ValueError, match="matched.*budget"):
        load_experiment(_write(tmp_path, mismatched))


def test_strict_types_hashes_and_formal_execution_controls(tmp_path):
    base = {
        "name": "formal", "stage": "online", "profile": "h200_formal", "formal": True,
        "model": {"name": "primary", "revision": "a" * 40},
        "benchmark": {"name": "taco_codecontests", "revision": "b" * 40, "digest": "sha256:" + "c" * 64},
        "baselines": ["goav_joint_aipw"], "seeds": [1, 2, 3], "expected_audit_fraction": 0.1,
        "bootstrap_replicates": 10000,
        "bank_manifest_hash": "sha256:" + "d" * 64,
        "gates": {"fixed_pool_required": True},
        "gate_spec_hash": "",
        "registered_config_hash": "",
        "abort": {"peak_memory_fraction": 0.9, "infrastructure_failure_fraction": 0.01, "cost_overrun_fraction": 0.25},
    }
    def sealed(value):
        value = dict(value)
        value["gate_spec_hash"] = canonical_hash(value["gates"])
        value["registered_config_hash"] = canonical_hash({key: item for key, item in value.items() if key != "registered_config_hash"})
        return value

    base = sealed(base)
    wrong_registration = {**base, "registered_config_hash": "sha256:" + "f" * 64}
    with pytest.raises(ValueError, match="registered config hash"):
        load_experiment(_write(tmp_path, wrong_registration))
    for field, value, message in (("bootstrap_replicates", 9999, "10000"), ("formal", 1, "boolean")):
        bad = dict(base)
        bad[field] = value
        bad = sealed(bad)
        with pytest.raises(ValueError, match=message):
            load_experiment(_write(tmp_path, bad))
    bad_digest = sealed({**base, "benchmark": {**base["benchmark"], "digest": "sha256:" + "z" * 64}})
    with pytest.raises(ValueError, match="sha256"):
        load_experiment(_write(tmp_path, bad_digest))


def test_pending_formal_template_plans_but_cannot_activate_execution(tmp_path):
    gates = {"fixed_pool_required": True}
    pending = {
        "name": "pending", "stage": "online", "profile": "h200_formal", "formal": True,
        "formal_activation": "pending",
        "model": {"name": "primary", "revision": "a" * 40},
        "benchmark": {"name": "taco_codecontests", "revision": "b" * 40, "digest": "sha256:" + "c" * 64},
        "baselines": ["goav_joint_aipw"], "seeds": [1, 2, 3], "expected_audit_fraction": 0.1,
        "bootstrap_replicates": 10000, "bank_manifest_hash": "unresolved",
        "gates": gates, "gate_spec_hash": canonical_hash(gates), "registered_config_hash": "unresolved",
        "abort": {"peak_memory_fraction": 0.9, "infrastructure_failure_fraction": 0.01, "cost_overrun_fraction": 0.25},
    }
    path = _write(tmp_path, pending)
    assert load_experiment(path).raw["formal_activation"] == "pending"
    with pytest.raises(ValueError, match="pending.*execution"):
        load_experiment(path, for_execution=True)


def test_formal_execution_requires_verified_registration_artifacts(tmp_path):
    gates = {"fixed_pool_required": True}
    active = {
        "name": "active", "stage": "frozen", "profile": "h200_formal", "formal": True,
        "formal_activation": "registered", "model": {"name": "primary", "revision": "a" * 40},
        "benchmark": {"name": "taco_codecontests", "revision": "b" * 40, "digest": "sha256:" + "c" * 64},
        "baselines": ["goav_joint_aipw"], "seeds": [1, 2, 3], "expected_audit_fraction": 0.1,
        "bootstrap_replicates": 10000, "bank_manifest_hash": "sha256:" + "d" * 64,
        "gates": gates, "gate_spec_hash": canonical_hash(gates),
        "abort": {"peak_memory_fraction": 0.9, "infrastructure_failure_fraction": 0.01, "cost_overrun_fraction": 0.25},
        "execution_artifacts": {
            "bank_manifest": {"path": "bank.json", "digest": "sha256:" + "d" * 64},
            "phase_gate": {"path": "gate.json", "digest": "sha256:" + "e" * 64},
            "evaluator_attestation": {"path": "attestation.json", "digest": "sha256:" + "f" * 64},
        },
    }
    active["registered_config_hash"] = canonical_hash(active)
    path = _write(tmp_path, active)
    assert load_experiment(path).formal
    with pytest.raises(ValueError, match="registration root"):
        load_experiment(path, for_execution=True)


def test_formal_execution_reads_and_verifies_gate_bank_and_attestation(tmp_path):
    from goav.manifest import file_digest
    root = tmp_path / "registration"
    root.mkdir()
    gates = {"fixed_pool_required": True}
    (root / "bank.json").write_text(json.dumps({"bank_hash": "sha256:" + "9" * 64}), encoding="utf-8")
    (root / "gate.json").write_text(json.dumps({"passed": True, "gate_spec_hash": canonical_hash(gates)}), encoding="utf-8")
    (root / "attestation.json").write_text(json.dumps({"status": "verified", "process_isolation": True, "restricted_sandbox": True, "network_disabled": True}), encoding="utf-8")
    artifacts = {name: {"path": filename, "digest": file_digest(root / filename)} for name, filename in (("bank_manifest", "bank.json"), ("phase_gate", "gate.json"), ("evaluator_attestation", "attestation.json"))}
    active = {
        "name": "active", "stage": "frozen", "profile": "h200_formal", "formal": True, "formal_activation": "registered",
        "model": {"name": "primary", "revision": "a" * 40}, "benchmark": {"name": "taco_codecontests", "revision": "b" * 40, "digest": "sha256:" + "c" * 64},
        "baselines": ["goav_joint_aipw"], "seeds": [1, 2, 3], "expected_audit_fraction": 0.1, "bootstrap_replicates": 10000,
        "bank_manifest_hash": artifacts["bank_manifest"]["digest"], "gates": gates, "gate_spec_hash": canonical_hash(gates),
        "abort": {"peak_memory_fraction": 0.9, "infrastructure_failure_fraction": 0.01, "cost_overrun_fraction": 0.25}, "execution_artifacts": artifacts,
    }
    active["registered_config_hash"] = canonical_hash(active)
    path = _write(tmp_path, active)
    with pytest.raises(ValueError, match="attestation signature verification.*fail-closed"):
        load_experiment(path, for_execution=True, registration_root=root)
    (root / "attestation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_experiment(path, for_execution=True, registration_root=root)
