# Artifacts and evidence status

Run directories are immutable after checksum indexing; the checksum file is
created exclusively and cannot be overwritten. Safe relative paths prevent
directory escape. `goav-run verify` rejects a changed digest, missing file,
unexpected file, or requested config/bank binding mismatch. A run stores config
hashes, model/dataset revisions, bank binding, design-before-outcome JSONL,
exact propensities, cost ledgers, seed/RNG stream and estimator-status metadata.

Status values are intentionally narrow:

- `planned_not_evidence`: validated configuration only.
- `deterministic_fake_smoke_only`: plumbing evidence, never performance.
- `executed_unverified_label_update_not_evidence`: the model and optimizer ran,
  but caller-supplied labels cannot establish the registered causal estimand.
- `prepared_bank_transformers_frozen_diagnostics`: a real model/bank path ran,
  but no formal-evidence claim follows without an activated registration and
  external evaluator attestation.
- `prepared_bank_deterministic_diagnostic_not_model_evidence`: prepared data ran
  through the deterministic backend only.

Report gradient nMSE/bias/cosine/sign agreement, ESS and weight quantiles,
support violations, joint NLL/Brier/ECE/covariance RMSE, expected and realized
budgets, Pass@1/5, tokens, executions, CPU/wall seconds and GPU hours. Abort
scaling above 90% peak memory, 1% infrastructure failure or 25% projected cost
overrun.

`ess_fraction` is Kish ESS divided by the number of included candidates on
non-empty audits, then averaged over eligible non-empty draws. It lies in
`(0,1]`; the 0.3 gate is therefore feasible. Empty draws do not masquerade as
zero-quality weights. Deterministic zero-propensity diagnostics such as top-k
serialize `predicted_risk: null` with an explicit unavailable status rather
than emitting non-finite JSON.

Checksum sealing alone does not create `sealed_final_evidence`; that status is
intentionally unavailable in this repository. Formal H200 templates remain
pending until their real bank, gate, and registration hashes are supplied, and
formal activation additionally requires an external trusted attestation
verifier not shipped here.
