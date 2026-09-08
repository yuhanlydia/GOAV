import json
import os
from pathlib import Path
import re
import subprocess
import sys
import ast

import pytest
import yaml

from goav.config import load_experiment
from goav.bank import CandidateGroup, CheapTest, TrustedSidecar, bank_hash, write_bank
from goav.registry import BASELINES, BENCHMARKS, MODELS

ROOT = Path(__file__).parents[1]


def test_model_and_benchmark_pins_cover_registered_matrix():
    model_files = sorted((ROOT / "configs/models").glob("*.yaml"))
    benchmark_files = sorted((ROOT / "configs/benchmarks").glob("*.yaml"))
    models = {row["name"]: row for row in map(lambda path: yaml.safe_load(path.read_text()), model_files)}
    benchmarks = {row["name"]: row for row in map(lambda path: yaml.safe_load(path.read_text()), benchmark_files)}
    assert set(models) == set(MODELS)
    assert set(benchmarks) == set(BENCHMARKS)
    assert all(re.fullmatch(r"[0-9a-f]{40}", row["revision"]) for row in models.values())
    assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", row["digest"]) for row in benchmarks.values())


def test_every_experiment_config_validates_and_plans_without_downloads():
    configs = sorted((ROOT / "configs/experiments").glob("*.yaml"))
    assert {path.stem for path in configs} == {"smoke", "frozen_16gb", "online_24gb", "online_h200_formal", "qwen3_replication", "deepseek_transfer"}
    profiles = set()
    all_arms = set()
    all_ablations = set()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for path in configs:
        config = load_experiment(path)
        if config.formal:
            assert config.raw["formal_activation"] == "pending"
            with pytest.raises(ValueError, match="pending.*execution"):
                load_experiment(path, for_execution=True)
        profiles.add(config.profile)
        all_arms.update(config.baselines)
        all_ablations.update(config.raw["ablations"])
        completed = subprocess.run([sys.executable, "-m", "goav", "plan", str(path)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        assert json.loads(completed.stdout)["downloads"] is False
    assert profiles == {"smoke", "16gb", "24gb", "h200_formal"}
    assert set(BASELINES) <= all_arms
    assert {"remove_gradient_influence", "diagonal_covariance", "independent_bernoulli", "factorized_poisson", "amortized_policy", "oracle_covariance", "imputation_ht_aipw", "propensity_perturbation", "sketch_size", "candidate_count", "audit_budget", "inclusion_floor", "cheap_test_count", "anchor_corruption", "candidate_permutation", "identity_fisher_optimizer_metric"} <= all_ablations


def test_scripts_docs_and_ci_expose_reproducible_status_semantics(tmp_path):
    scripts = {path.name for path in (ROOT / "scripts").glob("*.sh")}
    assert scripts == {"run_smoke.sh", "run_frozen_7b.sh", "run_online.sh"}
    script_text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "scripts").glob("*.sh"))
    assert "python -c" not in script_text
    assert "--bank-jsonl" in script_text and "online-update" in script_text
    assert (ROOT / ".github/workflows/ci.yml").is_file()
    docs = "\n".join((ROOT / f"docs/{name}.md").read_text(encoding="utf-8") for name in ("EXPERIMENTS", "BASELINES", "DATA", "ARTIFACTS"))
    for name in ("VPO", "UTRL", "B4", "noise-corrected GRPO", "CURE", "CodeT", "CoSPlay"):
        assert name in docs
    assert "no 7B performance gain is claimed" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "nc_grpo" not in "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "configs").rglob("*.yaml"))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(["bash", str(ROOT / "scripts/run_smoke.sh"), str(tmp_path / "smoke")], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert (tmp_path / "smoke/result.json").is_file()
    result = json.loads((tmp_path / "smoke/result.json").read_text())
    assert result["result"]["evidence_status"] == "connected_synthetic_smoke_not_phase_evidence"
    assert result["result"]["arms"][0]["design_events"] == 8 * 20
    assert "gradient_nmse" in result["result"]["arms"][0]


def test_trainer_modules_have_ci_enforced_sidecar_import_boundary():
    for name in ("trainer.py", "models.py", "gradient.py", "loss.py"):
        tree = ast.parse((ROOT / "src/goav" / name).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(item.endswith("bank") or "sidecar" in item.lower() for item in imports)


def test_cli_executes_a_prepared_bank_path_without_downloading(tmp_path):
    from goav.cli import main
    tests = tuple(CheapTest(f"{kind}-{i}", kind, f"test-{kind}-{i}") for kind, count in (("curated", 12), ("generated", 12), ("fault_targeted", 12), ("metamorphic", 8), ("control", 4)) for i in range(count))
    group = CandidateGroup("p", "c", tuple(f"candidate-{i}" for i in range(8)), tests, __import__("numpy").zeros((8, 48), dtype=int))
    bank_jsonl, bank_npz = tmp_path / "bank.jsonl", tmp_path / "bank.npz"
    write_bank(bank_jsonl, bank_npz, [group])
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps({"bank_hash": bank_hash([group]), "labels": [{"problem_id": "p", "checkpoint_id": "c", "values": [0, 1, 0, 1, 0, 1, 0, 1]}]}), encoding="utf-8")
    output = tmp_path / "run"
    config = tmp_path / "prepared-config.yaml"
    config_payload = yaml.safe_load((ROOT / "configs/experiments/smoke.yaml").read_text(encoding="utf-8"))
    config_payload["baselines"] = ["uniform_aipw"]
    config_payload["execution"]["design_draws"] = 1
    config.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    assert main(["prepare", str(config), "--output", str(output)]) == 0
    assert main(["run", str(config), "--output", str(output), "--bank-jsonl", str(bank_jsonl), "--bank-npz", str(bank_npz), "--trusted-sidecar", str(sidecar_path), "--backend", "deterministic"]) == 0
    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["result"]["bank_hash"] == bank_hash([group])
    assert payload["status"] == "prepared_bank_deterministic_diagnostic_not_model_evidence"
