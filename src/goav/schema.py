"""Frozen public configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    role: str


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    source: str
    role: str


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    paper: str
    repository: str
    revision: str
    license: str
    mode: str
    biased: bool = False
    provenance_status: str = "resolved"
    executable: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    profile: str
    formal: bool
    model: Mapping[str, Any]
    benchmark: Mapping[str, Any]
    baselines: tuple[str, ...]
    seeds: tuple[int, ...]
    expected_audit_fraction: float
    config_hash: str
    optional_dependencies: Mapping[str, bool] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
