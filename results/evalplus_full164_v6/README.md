# EvalPlus full-164 frozen 7B diagnostic (v6)

This directory records the completed Qwen2.5-Coder-7B frozen-bank diagnostic
run on a 16 GB RTX A4000. It is a real model and real EvalPlus run, but it is
**not sealed formal evidence**. The runner status is
`prepared_bank_transformers_frozen_diagnostics` and the evidence status is
`frozen_diagnostics_not_sealed_evidence`.

## Bound inputs

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- EvalPlus revision: `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`
- Tasks: all 164 HumanEval+ tasks
- Candidates: K=8 initial real Qwen candidates per task
- Design draws: 200 per task and arm; expected trusted-audit budget 0.8 candidates/task
- Outcome model: cross-fitted
- Bank hash: `sha256:971af6515727a8fc4ce53716f8db4ceb866db19af24a9e2996757f0f1568ba9f`
- Config hash: `sha256:9f290e9949d1429ab56d97a1c81be8a58ab6eff04bd21bf6360b3432f304964b`

## Result

| Arm | gradient nMSE | mean cosine | realized budget | ESS fraction |
|---|---:|---:|---:|---:|
| uniform_ht | 4.7586 | 0.6196 | 0.7934 | 1.0000 |
| uniform_aipw | 0.7722 | 0.5298 | 0.7934 | 1.0000 |
| entropy_aipw | 0.7361 | 0.5299 | 0.7962 | 0.9995 |
| coverage_kill_aipw | 0.5974 | 0.5294 | 0.7940 | 0.9680 |
| factorized_neyman_aipw | 0.7686 | 0.5298 | 0.7938 | 1.0000 |
| bayes_voi_aipw | 0.7546 | 0.5299 | 0.7941 | 1.0000 |
| goav_joint_aipw | 0.6711 | 0.5346 | 0.7989 | 0.9920 |

The preregistered nMSE gate did **not** pass in this run: the GOAV joint arm had aggregate nMSE 0.6711, and `nmse_reduction=-0.1235`. The ESS gate and support gate passed. This is reported as a negative diagnostic result, not a method claim.

## Integrity and limits

`goav verify` passed. The independent streaming ledger audit checked all seven
arms: each has 32,800 immutable design events and 32,800 matching outcome
events for 164 tasks and 200 draws/task, with unique design IDs, event pairing,
bank binding, draw ranges, expected budgets, and outcome cardinality. Raw event
logs are omitted from Git because they are large; their SHA-256 digests are
retained in `checksums.json` and independently recomputed in
`independent-ledger-audit.json`.

This diagnostic does not execute the registered three-checkpoint Phase-B pool,
cheap-test-count and candidate-permutation ablations, external evaluator
attestation, or the H200 stage. It is evidence that the full local diagnostic
chain ran and that this v6 protocol did not meet the preregistered positive
nMSE gate.
