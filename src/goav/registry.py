"""Preregistered model, benchmark and baseline registries."""

from __future__ import annotations

from .schema import BaselineSpec, BenchmarkSpec, ModelSpec


MODELS = {
    "plumbing": ModelSpec("plumbing", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "end-to-end smoke"),
    "primary": ModelSpec("primary", "Qwen/Qwen2.5-Coder-7B-Instruct", "frozen bank and online formal"),
    "architecture_replication": ModelSpec("architecture_replication", "Qwen/Qwen3-8B", "strongest-arm replication"),
    "cross_family": ModelSpec("cross_family", "deepseek-ai/deepseek-coder-6.7b-instruct", "frozen transfer"),
    "utrl_control": ModelSpec("utrl_control", "dgjun32/UTRL-4B", "learned-test control"),
    "reasonflux_4b": ModelSpec("reasonflux_4b", "Gen-Verse/ReasonFlux-Coder-4B", "learned-test control"),
    "reasonflux_7b": ModelSpec("reasonflux_7b", "Gen-Verse/ReasonFlux-Coder-7B", "learned-test control"),
}

BENCHMARKS = {
    "synthetic": BenchmarkSpec("synthetic", "builtin", "exact design identification"),
    "codecontests_o": BenchmarkSpec("codecontests_o", "CodeContests-O", "problem-disjoint frozen bank"),
    "taco_codecontests": BenchmarkSpec("taco_codecontests", "TACO+CodeContests", "online train/dev"),
    "evalplus": BenchmarkSpec("evalplus", "EvalPlus", "public mechanism replication"),
    "livecodebench_v6": BenchmarkSpec("livecodebench_v6", "LiveCodeBench/v6", "public evaluation"),
    "bigcodebench_hard": BenchmarkSpec("bigcodebench_hard", "BigCodeBench-Hard/Instruct", "cross-domain evaluation"),
    "held_out": BenchmarkSpec("held_out", "evaluator-sidecar", "locked final evidence"),
}


def _internal(name: str, paper: str, mode: str, *, biased: bool = False) -> BaselineSpec:
    return BaselineSpec(name, paper, "https://github.com/yuhanlydia/GOAV", "3ebfe4f", "repository-has-no-license-file", mode, biased, "resolved-local-protocol", True)


def _external(name: str, paper: str, repository: str, mode: str) -> BaselineSpec:
    # External comparator source archives are intentionally unresolved until a
    # formal registration pins and licenses the exact implementation used.
    return BaselineSpec(name, paper, repository, "unresolved-not-executable", "unverified-not-executable", mode, False, "unresolved-upstream-pin", False)


BASELINES = {
    "cheap_only": _internal("cheap_only", "GOAV controlled protocol", "controlled"),
    "uniform_ht": _internal("uniform_ht", "Horvitz-Thompson estimator (JASA 1952)", "controlled"),
    "uniform_aipw": _internal("uniform_aipw", "augmented inverse-probability weighting", "controlled"),
    "entropy_aipw": _internal("entropy_aipw", "entropy sampling protocol", "controlled"),
    "coverage_kill_aipw": _internal("coverage_kill_aipw", "coverage/kill sampling protocol", "controlled"),
    "factorized_neyman_aipw": _internal("factorized_neyman_aipw", "Neyman allocation protocol", "controlled"),
    "bayes_voi_aipw": _internal("bayes_voi_aipw", "Bayesian value-of-information protocol", "controlled"),
    "noise_corrected_grpo": _external("noise_corrected_grpo", "Noise-corrected GRPO", "https://github.com/omarito101/GRPO_project", "registered-unimplemented"),
    "noise_contrastive_grpo": _external("noise_contrastive_grpo", "Noise-contrastive GRPO paper specification", "unresolved-paper-source", "registered-unimplemented"),
    "goav_joint_aipw": _internal("goav_joint_aipw", "GOAV design protocol", "controlled"),
    "oracle_covariance_goav": _internal("oracle_covariance_goav", "GOAV oracle-covariance diagnostic", "diagnostic"),
    "full_audit": _internal("full_audit", "full-audit ceiling", "ceiling"),
    "top_k": _internal("top_k", "deterministic top-k diagnostic", "biased-control", biased=True),
    "grpo": _external("grpo", "DeepSeekMath (arXiv:2402.03300)", "https://github.com/deepseek-ai/DeepSeek-Math", "registered-unimplemented"),
    "dr_grpo": _external("dr_grpo", "Dr. GRPO comparator", "https://github.com/volcengine/verl", "registered-unimplemented"),
    "vpo": _external("vpo", "Vector Policy Optimization comparator", "unresolved-official-repository", "registered-unimplemented"),
    "b4": _external("b4", "B4 comparator", "unresolved-official-repository", "registered-unimplemented"),
    "codet": _external("codet", "CodeT (arXiv:2207.10397)", "https://github.com/microsoft/CodeT", "registered-unimplemented"),
    "cure": _external("cure", "CURE learned-test comparator", "unresolved-official-repository", "registered-unimplemented"),
    "utrl_4b": _external("utrl_4b", "UTRL frozen control", "https://github.com/dgjun32/UTRL", "registered-unimplemented"),
    "cosplay": _external("cosplay", "CoSPlay", "https://github.com/sanae-ai/CoSPlay", "registered-unimplemented"),
}
