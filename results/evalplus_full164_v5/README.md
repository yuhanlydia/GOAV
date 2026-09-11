# EvalPlus full-164 frozen 7B diagnostic (v5)

This directory records the completed Qwen2.5-Coder-7B frozen-bank diagnostic
run executed on a 16 GB RTX A4000. It is a real model and real EvalPlus run,
but it is **not sealed formal evidence**. The exact status emitted by the
runner is `prepared_bank_transformers_frozen_diagnostics`.

## Bound inputs

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Model revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- EvalPlus revision: `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`
- Tasks: all 164 HumanEval+ tasks
- Candidates: K=8 initial real Qwen candidates per task
- Cheap tests: 48 HumanEval+ inputs per candidate
- Trusted outcome: full HumanEval+ suite pass/fail
- Design draws: 200 per task and arm
- Expected trusted-audit budget: 0.8 candidates per task (10%)
- Bank hash: `sha256:971af6515727a8fc4ce53716f8db4ceb866db19af24a9e2996757f0f1568ba9f`
- Config hash: `sha256:0599263e4b1410edb7f5404edd0394a0d448a3d6e1d09162e9c4d32cc1f8a93f`

The run began from repository base `24080d5` with the memory-safe event,
microbatch, and 7B model patches in the working tree. Those patches, plus a
post-run statistically equivalent cluster-streaming refinement, are committed
as `0ae97abbf2c4d381c071cfc18b874756dc142c43`. Therefore `0ae97ab` is the
published implementation lineage, not a byte-identical sealed execution pin.

## Result

| Arm | gradient nMSE | mean cosine | realized budget | ESS fraction |
|---|---:|---:|---:|---:|
| uniform HT | 4.8845 | 0.6195 | 0.7934 | 1.0000 |
| uniform AIPW | 0.8528 | 0.4429 | 0.7934 | 1.0000 |
| entropy AIPW | 0.7405 | 0.4427 | 0.7927 | 0.9994 |
| coverage/kill AIPW | 0.7009 | 0.4412 | 0.7940 | 0.9680 |
| factorized Neyman AIPW | 0.8548 | 0.4429 | 0.7931 | 1.0000 |
| Bayes-VOI AIPW | 0.8535 | 0.4427 | 0.7930 | 0.9999 |
| GOAV joint AIPW | **0.5348** | **0.4445** | 0.7957 | 0.9969 |

GOAV reduced aggregate nMSE by 23.70% relative to the strongest aggregate
eligible comparator, coverage/kill AIPW. The preregistered statistical gate did
not pass: the Holm-adjusted paired randomization p-value against coverage/kill
was 0.2375, so `nmse_gate=false`. The ESS and support gates passed.

## Integrity

The runner checksum verifier passed after completion. An independent streaming
audit then checked all seven event logs: each contains 32,800 immutable design
events and 32,800 matching outcome events, for 229,600 audited draws overall.
It also checked event ordering, unique design IDs, bank bindings, draw ranges,
expected budgets, uniform variates, and revealed-outcome cardinalities.

The raw event logs are omitted from Git because they total 1.8 GB. Their exact
SHA-256 digests are retained in `checksums.json`; the independently recomputed
digests are in `independent-ledger-audit.json`. The canonical bank and
post-freeze trusted sidecar are included under `bank/`.

## Evidence limits and next audit

This diagnostic used one checkpoint and 164 tasks. It did not execute the
registered three-checkpoint Phase-B pool, the declared cheap-test-count and
candidate-permutation ablations, an external evaluator attestation, or the
online H200 stage. The current Transformers path also supplies a diagonal
plug-in covariance instead of wiring the repository's cross-fitted Ising
outcome model into the frozen runner. That missing method connection is being
debugged before any rerun or positive method claim.
