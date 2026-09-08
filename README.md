# GOAV

GOAV is a standalone, preregistered implementation of gradient-optimal active
verification for coding-RL experiments. Its CPU core implements exact joint
outcome moments, full-support audit designs and design-unbiased estimators. The
model integrations are optional and loaded only when explicitly requested.

At this preregistered implementation stage, no 7B performance gain is claimed.
See `docs/EXPERIMENTS.md`
for the evidence stages and status semantics once experiment configurations are
installed.

The repository does not provide a restricted code-execution sandbox or OS-level
evaluator isolation. Formal H200 configs are non-executable pending templates,
and external practical baselines with unresolved upstream pins fail closed.

Start with `python -m pip install -e '.[test]'`, `goav-run doctor`, and
`goav-run plan configs/experiments/smoke.yaml`. A download-free end-to-end check
is `bash scripts/run_smoke.sh artifacts/smoke`.
