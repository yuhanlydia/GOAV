# GOAV 7B Coding-RL Experiment Design

**Status:** preregistered implementation target; no 7B or active-verification gain is claimed.

## Claim and identification target

GOAV learns a randomized full-support trusted-audit design under correlated noisy cheap tests, with known first- and second-order inclusion propensities, to minimize clean policy-gradient estimation risk per cost. Fixed-pool auditing is the paper's identified first claim. Learned test content is a gated extension and cannot be mixed into the first claim.

For K=8 candidates, the theoretical path enumerates all 256 audit subsets. The clean target uses the trainer's exact token mask and loss denominator. AIPW labels are `mu_i + D_i/pi_i * (Y_i-mu_i)`. Propensities and the full subset-distribution hash are recorded before any trusted outcome is revealed.

Unbiasedness is claimed only for the fixed on-policy linear one-update estimand. PPO clipping, group normalization, multi-epoch reuse and vector/set objectives are labelled practical approximations.

## Repository boundary

The repository owns joint outcome/covariance models, score-gradient influence geometry, exact/amortized audit policies, HT/AIPW estimators, fixed candidate/test banks, leakage-proof evaluator sidecars, model/benchmark adapters, online QLoRA entry points and statistics. PBPF state inference and JAG trees do not belong here.

## Models

| Role | Hugging Face ID | Scope |
|---|---|---|
| plumbing | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | end-to-end smoke |
| primary | `Qwen/Qwen2.5-Coder-7B-Instruct` | frozen bank and online formal |
| architecture replication | `Qwen/Qwen3-8B` | strongest two-arm replication |
| cross-family | `deepseek-ai/deepseek-coder-6.7b-instruct` | frozen transfer |
| tester controls | `dgjun32/UTRL-4B`, ReasonFlux-Coder 4B/7B | learned-test extension only |

All model revisions resolve to commit SHAs and are reported separately.

## Benchmarks and roles

| Dataset | Role |
|---|---|
| synthetic correlated-oracle panel | exact design bias/risk identification |
| CodeContests-O | 1,500-task problem-disjoint fixed-policy bank, 1,000/250/250 |
| TACO + CodeContests | online RL train/dev pool |
| EvalPlus | transparent public mechanism replication |
| LiveCodeBench v6 | public evaluation; strict 2025-02-01 to 2025-05-01 slice reported separately |
| BigCodeBench-Hard Instruct | public practical/API cross-domain evaluation |
| evaluator-held post-freeze suite | one-shot locked final evidence |

Public LCB is not sealed for modern systems. Registered full-suite outcome is an operational grading target, not semantic truth. Splits and deduplication precede generation.

## Baselines and provenance

The controlled mechanism table shares rollout, cheap outcomes, policy loss and trusted budget: cheap/imputation only; uniform+HT; uniform+AIPW; entropy+AIPW; coverage/kill+AIPW; factorized Neyman+AIPW; Bayes-VOI+AIPW; noise-corrected GRPO with matched calibration; GOAV joint+AIPW; oracle-covariance GOAV; full audit ceiling; deterministic top-k as a known biased control.

The practical table separately compares GRPO/Dr.GRPO, VPO, noise-corrected GRPO, B4, CodeT, CURE, frozen UTRL-4B and CoSPlay with matched tokens/executions/CPU. CodeContests-O is data infrastructure, not an oracle. UTRL's released training code is incomplete and cannot be presented as an official 7B training reproduction.

Names are unambiguous: `noise_corrected_grpo` maps to `omarito101/GRPO_project`; `noise_contrastive_grpo` denotes a different paper-spec exploration. The alias `nc_grpo` is rejected. Every baseline stores paper/repo/revision/license/mode provenance.

## Joint model and audit policy

For K=8, a permutation-equivariant pairwise Ising model over candidate labels is enumerated exactly to obtain partition function, mean and covariance. Features include cheap execution patterns, origin, exception/timeout, coverage, gradient norms and pairwise cosine. Cross-fitting groups by problem and checkpoint; test candidate order is randomized to detect index leakage.

The exact policy enumerates audit subsets and recomputes `pi_i` and `pi_ij` by summation. It enforces expected audit budget and inclusion floor, mixes with symmetric exploration and fails closed on support or budget violations. The amortized policy is distilled from the exact solver and remains subject to exact propensity recomputation.

## Experiment stages

### A synthetic/exact mechanism

Use 256–512 tasks, real frozen score geometry, K=8, 200 independent design draws, budget fractions 5/10/20/40/100 and primary noise FN=.1, FP=.2, ICC=.6. Sweep ICC 0/.3/.6/.9 and instance-/policy-dependent noise. Verify AIPW bias and gradient nMSE.

Primary gate at 10% budget: at least 20% nMSE reduction and +.05 cosine over the strongest AIPW baseline; standardized bias below .05; ESS fraction at least .30; zero support violations; sketch/exact risk-ranking Spearman at least .70.

### B frozen 7B bank

For each task and three policy checkpoints, sample K=8 candidates and execute 48 cheap tests: 12 curated, 12 generated, 12 fault-targeted, 8 metamorphic and 4 duplicate/corrupt controls. All arms replay the same matrix. Full-suite labels reside only in evaluator sidecars. Use five-fold problem/checkpoint-grouped cross-fitting. Advance if real-suite nMSE falls at least 10%, predicted versus realized risk Spearman reaches .30 and exact versus empirical propensities differ by at most .005.

### C online RL

Use Qwen2.5-Coder-7B, 4k-task pilot then 8k–15k formal pool, 800–1,200 updates, K=8 and prompt batch 32/64. Execute eight cheap tests across all candidates, expected trusted audit 10%, and freeze/update the outcome model only at preregistered blocks. Only GOAV and the Phase-B strongest baseline receive three formal seeds.

Online success is either sealed Pass@1 at least 1.5 points higher with CI lower bound above zero, or Pass@1 non-inferior within .5 point while using at least 30% less trusted CPU.

### D learned tests

Only after fixed-pool gates pass, add CodeContests-O/UTRL/CURE/CoSPlay test proposals. Test-selection propensity and candidate-audit propensity are distinct logs. Generated tests never grade themselves.

## Ablations

- remove gradient influence L; full to diagonal covariance; joint to independent Bernoulli; joint subsets to factorized Poisson; exact to amortized policy; oracle to learned cross-fitted covariance.
- imputation versus HT versus AIPW; exact versus clipped/self-normalized/perturbed propensity (bias diagnostics).
- sketch 64/128/256/512 and exact registered-block gradient; K=4/8/16; budgets 5/10/20/40/100; floors .005/.02/.05; cheap tests 4/8/16/32/48; anchor corruption 0/1/5%.
- candidate permutation and index-leak test; identity metric W as primary and Fisher/optimizer metrics as secondary.

## Statistics, budgets and artifacts

The base task is the statistical unit. Formal online runs use three independent seeds and at least 10,000 task-paired/hierarchical bootstrap replicates; secondary tests use Holm correction. Report gradient nMSE/bias/cosine/sign agreement, one-step trusted improvement, ESS and weight quantiles, support violations, joint NLL/Brier/ECE/covariance RMSE, exact and realized budgets, Pass@1/5, tokens, test executions, CPU seconds, wall time and GPU hours.

Run artifacts are immutable and checksum-bound. The trusted sidecar cannot be imported by trainer modules. Each design event is logged before its audit outcome with full distribution hash, exact `pi_i/pi_ij`, RNG stream and budget. Selection/model fitting folds never share base problems.

## Hardware profiles

- `16gb`: 7B NF4/AWQ frozen generation/banks and outcome/design models; 1.5B online smoke only.
- `24gb`: 7B NF4 QLoRA rank 16/32, microbatch 1, accumulation 32 and short pilot only.
- `h200_formal`: BF16 LoRA rank 32/alpha 64 with rollout and actor/ref separated; formal seeds.
- verifier farm: 64–128 CPU cores, network off, 2–4GB/container, 6–10 second timeout and program-test hash cache.

Peak memory above 90%, infrastructure failures above 1% or projected cost overrun above 25% aborts scaling.

## Failure handling and tests

Config validation rejects `nc_grpo`, non-full-support theoretical designs, mismatched budgets, missing revisions/digests and hidden-label imports. Tests prove subset enumeration, exact first/second-order propensities, AIPW design unbiasedness, permutation equivariance, Ising moments against brute force, budget/floor feasibility, event-before-outcome ordering, model-loss gradient equivalence on a tiny torch model, cross-fit disjointness, tamper rejection and deterministic fake-backend integration. GPU and external-data tests are opt-in; CPU CI downloads nothing.

