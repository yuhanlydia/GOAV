"""Download-free JSONL and lazy Hugging Face benchmark adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    prompt: str
    metadata: dict[str, Any]


class JsonlBenchmark:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[BenchmarkTask]:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            yield BenchmarkTask(str(row["task_id"]), str(row["prompt"]), {key: value for key, value in row.items() if key not in {"task_id", "prompt"}})


HF_BENCHMARKS = {
    "codecontests_o": ("open-r1/codecontests", None),
    "taco_codecontests": ("BAAI/TACO", "ALL"),
    "evalplus": ("evalplus/humanevalplus", None),
    "livecodebench_v6": ("livecodebench/code_generation_lite", "v6"),
    "bigcodebench_hard": ("bigcode/bigcodebench", "hard"),
}


def load_hf_benchmark(name: str, *, revision: str, split: str, streaming: bool = False):
    if name not in HF_BENCHMARKS:
        raise ValueError(f"unknown Hugging Face benchmark adapter: {name}")
    if not revision:
        raise ValueError("a pinned dataset revision is required")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for remote benchmark adapters; install goav[model]") from exc
    path, subset = HF_BENCHMARKS[name]
    return load_dataset(path, subset, revision=revision, split=split, streaming=streaming, trust_remote_code=False)


def load_codecontests_o(*, revision: str, split: str, streaming: bool = False):
    return load_hf_benchmark("codecontests_o", revision=revision, split=split, streaming=streaming)


def load_evalplus(*, revision: str, split: str = "test", streaming: bool = False):
    return load_hf_benchmark("evalplus", revision=revision, split=split, streaming=streaming)


def load_livecodebench_v6(*, revision: str, split: str = "test", streaming: bool = False):
    return load_hf_benchmark("livecodebench_v6", revision=revision, split=split, streaming=streaming)


def load_bigcodebench_hard(*, revision: str, split: str = "test", streaming: bool = False):
    return load_hf_benchmark("bigcodebench_hard", revision=revision, split=split, streaming=streaming)
