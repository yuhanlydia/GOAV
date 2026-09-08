# GOAV 7B Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone GOAV package for exact active-audit identification, frozen 7B bank replay and online QLoRA integration.

**Architecture:** The CPU core owns joint label moments, all-subset designs, exact propensities and HT/AIPW estimators. Strict sidecars and event ordering prevent outcome leakage; optional model adapters create candidate/gradient banks and practical online runs.

**Tech Stack:** Python 3.11/3.12, NumPy, SciPy, PyYAML; optional PyTorch, Transformers, PEFT, bitsandbytes, datasets.

**Spec:** `docs/superpowers/specs/2026-09-08-goav-7b-experiments-design.md`

## Global Constraints

- Package/CLI names are `goav` and `goav-run`; JAG/PBPF code is absent.
- Theoretical designs have full support, exact first/second-order propensities and design-before-outcome logs.
- `nc_grpo` is invalid; use `noise_corrected_grpo` or `noise_contrastive_grpo` explicitly.
- Formal configs pin all revisions/digests and require three seeds; local profiles are pilots.
- No smoke/fake result is described as a 7B performance result.

---

### Task 1: Package, registries and strict configuration

**Files:** create `pyproject.toml`, `README.md`, `src/goav/{__init__,__main__,cli,config,registry,schema}.py`, `tests/test_registry.py`.

**Interfaces:** produce frozen model/dataset/baseline specs, `load_experiment`, `validate_experiment`, and `goav-run doctor|plan`.

- [ ] Write a failing test covering exact models/data/arms/provenance, rejecting `nc_grpo`, missing revisions and formal-local profile combinations.
- [ ] Implement canonical YAML resolution/hash and lazy dependency checks without torch/data imports.
- [ ] Run tests and commit `feat: scaffold strict GOAV experiments`.

### Task 2: Candidate/test bank, sidecar and audit event protocol

**Files:** create `src/goav/{manifest,bank,ledger,sandbox,events}.py`, `tests/test_data_protocol.py`.

**Interfaces:** produce `CandidateGroup`, `CheapTest`, `TrustedSidecar`, `trainer_view`, `AuditDesignEvent`, `AuditOutcomeEvent` and immutable bank verification.

- [ ] Write failing tests for problem/checkpoint split grouping, inaccessible trusted labels, 12/12/12/8/4 test composition, event-before-outcome order, three cost ledgers and artifact tampering.
- [ ] Implement canonical JSONL/NPZ banks, checksums, strict trainer/evaluator types and deterministic fake sandbox.
- [ ] Run tests and commit `feat: add GOAV audit data protocol`.

### Task 3: Joint outcome model and cross-fitting

**Files:** create `src/goav/{outcomes,ising,crossfit,metrics}.py`, `tests/test_joint_model.py`.

**Interfaces:** produce `enumerate_binary`, `ising_moments`, `fit_fold`, `crossfit_predictions`, joint NLL/Brier/ECE/covariance RMSE.

- [ ] Write failing brute-force tests for partition/mean/covariance, candidate permutation equivariance and problem/checkpoint fold disjointness.
- [ ] Implement stable exact enumeration for K at most 16 and a CPU reference fitter; expose learned torch model as optional interface.
- [ ] Run tests and commit `feat: implement GOAV joint outcome models`.

### Task 4: Audit designs, estimators and controlled baselines

**Files:** create `src/goav/{subsets,design,estimators,baselines,audit}.py`, `tests/test_design.py`, `tests/test_estimators.py`.

**Interfaces:** produce `subset_matrix`, `inclusion_probabilities`, `design_risk`, `solve_design`, `ht_labels`, `aipw_labels`, `clean_gradient` and `run_design_audit`.

- [ ] Write failing exact tests for all 256 K=8 subsets, `pi_i/pi_ij`, budget/floor feasibility, AIPW unbiasedness, permutation invariance and empty-subset support.
- [ ] Implement uniform/entropy/coverage-kill/Neyman/Bayes-VOI/joint/oracle/full/top-k designs through one interface; tag top-k biased.
- [ ] Verify Monte Carlo propensity tolerance and commit `feat: implement gradient-optimal audits`.

### Task 5: Model gradient alignment and staged runners

**Files:** create `src/goav/{models,benchmarks,gradient,loss,experiment,trainer,statistics,artifacts}.py`, `tests/test_model_contract.py`, `tests/test_experiment.py`.

**Interfaces:** produce `PolicyBackend`, `score_geometry`, `token_mean_loss`, `run_frozen_bank`, `run_online`, and hierarchical bootstrap.

- [ ] Write failing tiny-torch test proving `L@Y` equals direct clean token-mean gradient and failing fake-backend tests for hidden labels/common banks/matched budgets.
- [ ] Implement lazy Qwen/Qwen3/DeepSeek adapters, JSONL plus CodeContests-O/EvalPlus/LCB/BigCodeBench adapters, frozen replay and optional QLoRA one-update path.
- [ ] Enforce theoretical one-update/unclipped status versus practical approximation metadata.
- [ ] Run tests and commit `feat: add model-backed GOAV experiments`.

### Task 6: Config matrix, documentation and CI

**Files:** create `configs/{models,benchmarks,experiments}/`, `scripts/{run_smoke,run_frozen_7b,run_online}.sh`, `docs/{EXPERIMENTS,BASELINES,DATA,ARTIFACTS}.md`, `.github/workflows/ci.yml`, `tests/test_repository_contract.py`; modify `README.md`.

**Interfaces:** provide exact pull/install/doctor/prepare/run/verify commands and evidence/status semantics.

- [ ] Write a failing repository-contract test enumerating required models, datasets, baselines, ablations, profiles, provenance and no-claim language.
- [ ] Add smoke/16GB/24GB/H200 configs and test every config through `goav-run plan` without downloads.
- [ ] Document VPO/UTRL/B4/noise-corrected-GRPO/CURE/CodeT/CoSPlay roles and official-code limitations.
- [ ] Run `pytest -q`, compileall, `git diff --check`, shell syntax and fake end-to-end smoke; commit `docs: document reproducible GOAV experiments`.

