"""Matched resource accounting for experiment arms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostLedger:
    tokens: int = 0
    test_executions: int = 0
    cpu_seconds: float = 0.0

    def charge(self, *, tokens: int = 0, test_executions: int = 0, cpu_seconds: float = 0.0) -> None:
        if tokens < 0 or test_executions < 0 or cpu_seconds < 0:
            raise ValueError("cost charges must be non-negative")
        self.tokens += int(tokens)
        self.test_executions += int(test_executions)
        self.cpu_seconds += float(cpu_seconds)

    def as_dict(self) -> dict[str, int | float]:
        return {"tokens": self.tokens, "test_executions": self.test_executions, "cpu_seconds": self.cpu_seconds}
