"""Command-line entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .artifacts import ArtifactStore, verify_checksum_index, write_checksum_index
from .config import dependency_status, load_experiment
from .bank import bank_hash, read_bank, read_trusted_sidecar
from .experiment import run_frozen_bank, run_online, synthetic_correlated_panel
from .models import DeterministicPolicyBackend, load_registered_model
from .manifest import file_digest


def _registered_bank_hash(config, registration_root: str | None) -> str | None:
    if not config.formal:
        return None
    record = config.raw["execution_artifacts"]["bank_manifest"]
    payload = json.loads((Path(registration_root) / record["path"]).read_text(encoding="utf-8"))
    value = payload.get("bank_hash")
    if type(value) is not str or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError("registered bank manifest does not contain a valid bank hash")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goav-run")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report optional dependency availability")
    plan = sub.add_parser("plan", help="validate and summarize a configuration")
    plan.add_argument("config")
    prepare = sub.add_parser("prepare", help="materialize a checksum-bound execution plan")
    prepare.add_argument("config")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--registration-root")
    run = sub.add_parser("run", help="execute a prepared download-free smoke configuration")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    run.add_argument("--bank-jsonl")
    run.add_argument("--bank-npz")
    run.add_argument("--trusted-sidecar")
    run.add_argument("--backend", choices=("deterministic", "transformers"))
    run.add_argument("--outcome-model", choices=("plugin", "crossfit"))
    run.add_argument("--registration-root")
    online = sub.add_parser("online-update", help="execute one verified causal-LM optimizer update")
    online.add_argument("config")
    online.add_argument("--batch-npz", required=True)
    online.add_argument("--bank-hash", required=True)
    online.add_argument("--algorithm", required=True)
    online.add_argument("--output", required=True)
    online.add_argument("--registration-root")
    verify = sub.add_parser("verify", help="verify an immutable run artifact directory")
    verify.add_argument("output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps({"cpu_core": True, "optional": dependency_status(), "evaluator_isolation": False, "restricted_sandbox": False}, sort_keys=True))
        return 0
    if args.command == "verify":
        verify_checksum_index(args.output)
        print(json.dumps({"verified": True, "output": str(Path(args.output).resolve())}, sort_keys=True))
        return 0
    config = load_experiment(args.config, for_execution=args.command in {"prepare", "run", "online-update"}, registration_root=getattr(args, "registration_root", None))
    if args.command == "prepare":
        store = ArtifactStore(args.output)
        store.write_json("plan.json", {"config": config.raw, "config_hash": config.config_hash, "status": "planned_not_evidence"})
        print(json.dumps({"prepared": True, "config_hash": config.config_hash, "output": str(Path(args.output).resolve())}, sort_keys=True))
        return 0
    if args.command == "run":
        execution = config.raw.get("execution", {})
        output = Path(args.output)
        if not (output / "plan.json").is_file():
            raise ValueError("prepare must be run before run")
        plan_payload = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        if plan_payload.get("config_hash") != config.config_hash:
            raise ValueError("prepared plan config hash mismatch")
        prepared_arguments = (args.bank_jsonl, args.bank_npz, args.trusted_sidecar)
        if any(prepared_arguments) and not all(prepared_arguments):
            raise ValueError("bank JSONL, bank NPZ and trusted sidecar must be provided together")
        if all(prepared_arguments):
            groups = read_bank(args.bank_jsonl, args.bank_npz)
            prepared_hash = bank_hash(groups)
            sidecar = read_trusted_sidecar(args.trusted_sidecar, expected_bank_hash=prepared_hash)
            if config.formal and _registered_bank_hash(config, args.registration_root) != prepared_hash:
                raise ValueError("formal bank manifest hash does not match prepared bank")
            backend_name = args.backend or "transformers"
            if backend_name == "deterministic":
                backend = DeterministicPolicyBackend()
                status = "prepared_bank_deterministic_diagnostic_not_model_evidence"
            else:
                backend = load_registered_model(config.model["name"], config.model["revision"], quantization=config.model.get("quantization"))
                status = "prepared_bank_transformers_frozen_diagnostics"
        elif execution.get("backend") == "fake" and args.backend in {None, "deterministic"}:
            groups, sidecar = synthetic_correlated_panel(int(execution.get("tasks", 8)), seed=config.seeds[0])
            backend = DeterministicPolicyBackend()
            status = "deterministic_fake_smoke_only"
        else:
            raise ValueError("real frozen execution requires a prepared bank JSONL/NPZ and trusted sidecar")
        default_outcome_model = "crossfit" if all(prepared_arguments) and args.backend == "transformers" else "plugin"
        outcome_model = args.outcome_model or execution.get("outcome_model", default_outcome_model)
        result = run_frozen_bank(groups, sidecar, backend, config.baselines, expected_budget=8 * config.expected_audit_fraction, inclusion_floor=0.02, seed=config.seeds[0], event_directory=output / "events", design_draws=int(execution.get("design_draws", 1)), statistics_replicates=int(config.raw.get("bootstrap_replicates", 1_000)), outcome_model=outcome_model)
        ArtifactStore(output).write_json("result.json", {"status": status, "outcome_model": outcome_model, "config_hash": config.config_hash, "result": asdict(result)})
        write_checksum_index(output, config_hash=config.config_hash, bank_hash=result.bank_hash)
        print(json.dumps({"completed": True, "status": status, "output": str(output.resolve())}, sort_keys=True))
        return 0
    if args.command == "online-update":
        from .trainer import OnlineTrainingBatch, attach_lora
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for online execution; install goav[model]") from exc
        model_backend = load_registered_model(config.model["name"], config.model["revision"], quantization=config.model.get("quantization"))
        model_backend.model = attach_lora(model_backend.model, rank=int(config.model.get("lora_rank", 32)), alpha=int(config.model.get("lora_alpha", 64)))
        device = next(model_backend.model.parameters()).device
        with np.load(args.batch_npz, allow_pickle=False) as batch:
            if set(batch.files) != {"input_ids", "labels", "response_mask"}:
                raise ValueError("online batch NPZ must contain input_ids, labels and response_mask")
            training_batch = OnlineTrainingBatch(torch.as_tensor(batch["input_ids"], dtype=torch.long, device=device), torch.as_tensor(batch["labels"], dtype=torch.float32, device=device), torch.as_tensor(batch["response_mask"], dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW((parameter for parameter in model_backend.model.parameters() if parameter.requires_grad), lr=float(config.raw.get("learning_rate", 1e-5)))
        if config.formal and _registered_bank_hash(config, args.registration_root) != args.bank_hash:
            raise ValueError("online bank hash does not match registered bank manifest")
        result = run_online(model_backend, algorithm=args.algorithm, training_batch=training_batch, optimizer=optimizer, claim_identified=False)
        output = Path(args.output)
        store = ArtifactStore(output)
        store.write_json("online-result.json", {"status": result.status, "config_hash": config.config_hash, "batch_hash": file_digest(args.batch_npz), "result": asdict(result)})
        write_checksum_index(output, config_hash=config.config_hash, bank_hash=args.bank_hash)
        print(json.dumps({"completed": True, "status": result.status, "output": str(output.resolve())}, sort_keys=True))
        return 0
    print(json.dumps({"name": config.name, "profile": config.profile, "formal": config.formal, "config_hash": config.config_hash, "downloads": False}, sort_keys=True))
    return 0
