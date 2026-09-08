# Baselines and provenance

The controlled mechanism table shares rollouts, cheap outcomes, trainer loss,
trusted budget and exact event protocol. It contains cheap-only/imputation,
uniform HT/AIPW, entropy AIPW, coverage/kill AIPW, factorized Neyman AIPW,
Bayes-VOI AIPW, noise-corrected GRPO, GOAV joint AIPW, oracle-covariance GOAV,
the full-audit ceiling and deterministic top-k as an explicitly biased control.

`noise_corrected_grpo` means the `omarito101/GRPO_project` method.
`noise_contrastive_grpo` is a separate paper-spec exploration. The short alias
is deliberately rejected by configuration and runtime validation.

The separate practical table includes GRPO/Dr.GRPO, VPO, B4, CodeT, CURE,
frozen UTRL-4B and CoSPlay under matched tokens, executions and CPU. VPO, B4,
CodeT and CoSPlay are external practical comparators, not redefinitions of the
GOAV estimand. CURE and ReasonFlux are learned-test extensions only. UTRL is a
frozen control: its released training repository does not constitute a complete
official 7B training reproduction, so GOAV does not label it as one.

Every baseline registry entry records paper, repository, revision, license,
mode, provenance status, and executability. The locally implemented audit
controls point to this repository protocol. External practical comparators are
explicitly `unresolved-upstream-pin` and non-executable here until a formal
registration supplies an exact source archive/commit, verified license, and
local patch digest. Formal execution rejects unresolved entries; a name in the
matrix is not a claim that its implementation is present.

Only entropy, coverage/kill, factorized Neyman, and Bayes-VOI AIPW are eligible
non-GOAV adaptive controls for the primary nMSE gate. Full audit and oracle are
diagnostics, top-k is biased, and uniform designs are non-adaptive; none enters
that comparator minimum.
