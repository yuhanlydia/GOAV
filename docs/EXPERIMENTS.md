# Experiments

GOAV separates evidence into four stages. A synthetic exact stage identifies
design bias and risk; a frozen 7B bank replays common candidate/test matrices;
online QLoRA is a practical staged experiment; learned tests begin only after
the fixed-pool gates pass. A fake smoke run validates plumbing and is never a
model-quality result.

## Reproducible commands

```bash
git clone https://github.com/yuhanlydia/GOAV.git goav
cd goav
python -m pip install -e '.[test]'
goav-run doctor
goav-run plan configs/experiments/frozen_16gb.yaml
goav-run prepare configs/experiments/smoke.yaml --output artifacts/smoke
goav-run run configs/experiments/smoke.yaml --output artifacts/smoke
goav-run verify artifacts/smoke
```

The smoke wrapper performs the last four operations with
`bash scripts/run_smoke.sh artifacts/smoke`. A prepared frozen bank is executed
with `scripts/run_frozen_7b.sh CONFIG BANK_JSONL BANK_NPZ TRUSTED_SIDECAR OUTPUT`.
One real causal-LM optimizer update (explicitly non-evidence because its caller
labels lack sealed audit provenance) is executed with
`scripts/run_online.sh CONFIG BATCH_NPZ BANK_HASH ALGORITHM OUTPUT`; the NPZ
must contain `input_ids`, candidate-level `labels`, and the shifted
`response_mask`. These commands load Transformers/PEFT lazily and may download
the pinned model only when explicitly run. Planning and CPU CI never download.

The H200 YAML files are preregistration templates with
`formal_activation: pending`, `bank_manifest_hash: unresolved`, and
`registered_config_hash: unresolved`. They can be planned but cannot be
prepared or run. Activation requires a real sealed bank hash, resolved upstream
comparator provenance, gate-artifact bindings, and a recomputed canonical
registration hash. This repository has no trusted attestation-verifier key, so
even internally consistent registration artifacts fail closed after checksum
and content validation; an external evaluator integration is required.

## Stage contracts

- Synthetic: 256–512 tasks, K=8, 200 design draws, audit fractions
  5/10/20/40/100%, FN=.1, FP=.2 and ICC=.6 primary.
- Frozen: CodeContests-O split 1,000/250/250 before generation, three policy
  checkpoints, K=8, 48 fixed cheap tests and five cross-fit folds grouped by
  base problem or near-duplicate cluster across every checkpoint.
- Online: 4k pilot then 8k–15k formal tasks, 800–1,200 updates, prompt batch
  32/64, eight common cheap tests and 10% expected trusted auditing.
- Learned tests: selection and audit propensities are separate; a generated
  test never grades itself.

Only an unclipped, on-policy, linear one-update estimator has identified status.
Clipping, group normalization, multi-epoch reuse and vector/set objectives are
outside the connected identified runner and remain `planned_not_evidence`.
Formal runs require three independent seeds and at least 10,000 hierarchical
bootstrap replicates.

The K=8 joint design is a constrained local SLSQP optimizer over all 256 subset
probabilities. It evaluates the declared exact `design_risk`, recomputes
`pi_i/pi_ij`, checks floor/budget/support feasibility, and reports convergence
and feasible-baseline risks. It does not claim a certified global optimum.
Frozen comparisons replay one registered uniform variate per task/draw through
each arm's inverse CDF. The nMSE gate considers only preregistered unbiased
adaptive non-GOAV controls; ceilings, oracle diagnostics, uniform sampling and
biased top-k are excluded. Paired task-cluster tests are Holm corrected.
Standardized bias divides the MC mean bias by a task-clustered standard error,
not by raw residual dispersion.
