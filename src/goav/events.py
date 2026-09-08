"""Append-only audit design/outcome event protocol."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import canonical_hash
from .subsets import inclusion_probabilities, subset_matrix


@dataclass(frozen=True)
class AuditDesignEvent:
    run_id: str
    bank_hash: str
    problem_id: str
    checkpoint_id: str
    group_hash: str
    candidate_order_hash: str
    draw_index: int
    design_id: str
    subset_probabilities: tuple[float, ...]
    pi_i: tuple[float, ...]
    pi_ij: tuple[tuple[float, ...], ...]
    rng_stream: str
    expected_budget: float
    distribution_hash: str
    uniform_draw: float | None = None

    @classmethod
    def create(cls, run_id: str, problem_id: str, checkpoint_id: str, design_id: str, subset_probabilities: Sequence[float], pi_i: Sequence[float], pi_ij: Sequence[Sequence[float]], rng_stream: str, *, bank_hash: str, group_hash: str, candidate_order_hash: str, draw_index: int, uniform_draw: float | None = None) -> "AuditDesignEvent":
        probabilities = tuple(float(x) for x in subset_probabilities)
        first_order = tuple(float(x) for x in pi_i)
        event = cls(run_id, bank_hash, problem_id, checkpoint_id, group_hash, candidate_order_hash, int(draw_index), design_id, probabilities, first_order, tuple(tuple(float(x) for x in row) for row in pi_ij), rng_stream, float(sum(first_order)), canonical_hash(probabilities), None if uniform_draw is None else float(uniform_draw))
        event.validate()
        return event

    def validate(self) -> None:
        for label, value in (("bank", self.bank_hash), ("group", self.group_hash), ("candidate order", self.candidate_order_hash), ("distribution", self.distribution_hash)):
            if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
                raise ValueError(f"{label} hash must be sha256")
        if self.draw_index < 0 or not self.rng_stream:
            raise ValueError("draw identity and RNG stream are required")
        if self.uniform_draw is not None and (not math.isfinite(self.uniform_draw) or not 0 <= self.uniform_draw < 1):
            raise ValueError("registered uniform draw must be finite and in [0, 1)")
        probabilities = np.asarray(self.subset_probabilities, dtype=float)
        if probabilities.ndim != 1 or len(probabilities) == 0 or len(probabilities) & (len(probabilities) - 1) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
            raise ValueError("event subset distribution is invalid")
        if canonical_hash(tuple(map(float, probabilities))) != self.distribution_hash:
            raise ValueError("event distribution hash mismatch")
        subsets = subset_matrix(int(np.log2(len(probabilities))))
        recomputed_i, recomputed_ij = inclusion_probabilities(subsets, probabilities)
        if not np.allclose(recomputed_i, self.pi_i, atol=1e-10, rtol=1e-10):
            raise ValueError("event first-order propensities mismatch")
        if not np.allclose(recomputed_ij, self.pi_ij, atol=1e-10, rtol=1e-10):
            raise ValueError("event second-order propensities mismatch")
        if not np.isclose(recomputed_i.sum(), self.expected_budget, atol=1e-10):
            raise ValueError("event expected budget mismatch")


@dataclass(frozen=True)
class AuditOutcomeEvent:
    run_id: str
    problem_id: str
    design_id: str
    audited: tuple[int, ...]
    outcomes: tuple[float, ...]


class AuditEventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _write(self, kind: str, event: AuditDesignEvent | AuditOutcomeEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event_type": kind, **asdict(event)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def append_design(self, event: AuditDesignEvent) -> None:
        event.validate()
        if any(isinstance(item, AuditDesignEvent) and item.design_id == event.design_id for item in self.read()):
            raise RuntimeError("audit design is immutable")
        self._write("design", event)

    @staticmethod
    def _validate_outcome(event: AuditOutcomeEvent, prior: list[AuditDesignEvent | AuditOutcomeEvent]) -> None:
        matching = [item for item in prior if isinstance(item, AuditDesignEvent) and item.run_id == event.run_id and item.problem_id == event.problem_id and item.design_id == event.design_id]
        if not matching:
            raise RuntimeError("design must be committed before outcome")
        if any(isinstance(item, AuditOutcomeEvent) and item.design_id == event.design_id for item in prior):
            raise RuntimeError("audit outcome is immutable")
        if len(event.audited) != len(matching[0].pi_i) or any(value not in (0, 1) for value in event.audited):
            raise ValueError("audit mask does not match committed design")
        design_event = matching[0]
        if design_event.uniform_draw is None:
            raise ValueError("audit outcome requires a committed uniform draw")
        probabilities = np.asarray(design_event.subset_probabilities, dtype=float)
        subset_index = min(int(np.searchsorted(np.cumsum(probabilities), design_event.uniform_draw, side="right")), len(probabilities) - 1)
        expected_mask = tuple(map(int, subset_matrix(len(design_event.pi_i))[subset_index]))
        if tuple(event.audited) != expected_mask:
            raise ValueError("audit mask is inconsistent with committed uniform draw")
        if len(event.outcomes) != sum(event.audited):
            raise ValueError("outcome event must contain exactly the revealed labels")
        if not all(math.isfinite(value) for value in event.outcomes):
            raise ValueError("audit outcomes must be finite")

    def append_outcome(self, event: AuditOutcomeEvent) -> None:
        prior = self.read()
        self._validate_outcome(event, prior)
        self._write("outcome", event)

    def read(self) -> list[AuditDesignEvent | AuditOutcomeEvent]:
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            kind = payload.pop("event_type")
            if kind == "design":
                payload["subset_probabilities"] = tuple(payload["subset_probabilities"])
                payload["pi_i"] = tuple(payload["pi_i"])
                payload["pi_ij"] = tuple(tuple(row) for row in payload["pi_ij"])
                event = AuditDesignEvent(**payload)
                event.validate()
                events.append(event)
            else:
                payload["audited"] = tuple(payload["audited"])
                payload["outcomes"] = tuple(payload["outcomes"])
                event = AuditOutcomeEvent(**payload)
                self._validate_outcome(event, events)
                events.append(event)
        return events
