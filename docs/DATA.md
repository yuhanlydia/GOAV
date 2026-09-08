# Data and evaluator firewall

Problem/checkpoint identifies a bank row, but cross-fitting assigns every
checkpoint of a base problem (or near-duplicate cluster) to the same fold.
The code asserts train/test base-identity disjointness. Splitting and
near-duplicate filtering happen before generation. Candidate order is
randomized at evaluation time to expose index leakage. CodeContests-O is fixed
data infrastructure, not an oracle.

Each frozen group has eight candidates and 48 cheap tests: 12 curated, 12
generated, 12 fault-targeted, eight metamorphic and four duplicate/corrupt
controls. All arms replay the identical matrix. JSONL metadata and NPZ arrays
are canonicalized and checksum-bound.

Trusted full-suite labels live in `TrustedSidecar`, never in `TrainerCandidateView`.
The evaluator commits the complete subset distribution hash, exact first- and
second-order propensities, shared uniform draw/RNG stream and expected budget
before requesting any trusted label. The outcome event contains only labels at
audited indices and is appended afterward; empty audits remain finite.

`TrustedSidecar` plus the CI import-boundary test is an in-process API/type
firewall, not a security boundary. This repository does not provide OS process
isolation, network restriction, syscall filtering, or time-limited code
execution. `FakeSandbox` is only a deterministic test double. A deployment must
supply and attest an external restricted evaluator before results can be formal
evidence. Frozen diagnostics measure evaluator orchestration with process CPU
time and label that provenance explicitly.

The frozen runner constructs and freezes every backend analysis and audit
design before its first sidecar call. Each draw event is then committed before
the sidecar reveals observed labels; event-log append recomputes the inverse-CDF
mask from the committed uniform variate. Full labels are read only afterward by
the evaluator for diagnostic target geometry and never feed back into designs.

LiveCodeBench v6 is public rather than sealed; the 2025-02-01 through
2025-05-01 slice is reported separately. The post-freeze evaluator-held suite
is one-shot locked final evidence.
