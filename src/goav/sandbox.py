"""Deterministic, download-free sandbox backend for integration tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .ledger import CostLedger


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    output_hash: str
    cpu_seconds: float


class FakeSandbox:
    security_boundary = "none_deterministic_test_double"

    def __init__(self, ledger: CostLedger):
        self.ledger = ledger

    def run(self, program: str, test: str) -> SandboxResult:
        raw = (program + "\0" + test).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        seconds = 0.001 + (int(digest[:6], 16) % 1000) / 1_000_000
        result = SandboxResult(int(digest[-2:], 16) % 2 == 0, "sha256:" + digest, seconds)
        self.ledger.charge(tokens=len(program.split()) + len(test.split()), test_executions=1, cpu_seconds=seconds)
        return result
