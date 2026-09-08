"""Evaluator-only orchestration of a committed audit draw."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bank import CandidateGroup, TrustedSidecar, candidate_group_payload
from .config import canonical_hash
from .design import AuditDesign
from .events import AuditDesignEvent, AuditEventLog, AuditOutcomeEvent


@dataclass(frozen=True)
class AuditResult:
    audited: np.ndarray
    revealed_indices: np.ndarray
    revealed_labels: np.ndarray
    design_id: str

    def __post_init__(self) -> None:
        audited = np.array(self.audited, dtype=np.int64, copy=True)
        indices = np.array(self.revealed_indices, dtype=np.int64, copy=True)
        labels = np.array(self.revealed_labels, dtype=float, copy=True)
        if audited.ndim != 1 or not np.isin(audited, [0, 1]).all() or not np.array_equal(indices, np.flatnonzero(audited)):
            raise ValueError("revealed indices must exactly match the audit mask")
        if labels.shape != indices.shape or not np.isfinite(labels).all():
            raise ValueError("revealed labels must be finite and align with indices")
        for value in (audited, indices, labels):
            value.flags.writeable = False
        object.__setattr__(self, "audited", audited)
        object.__setattr__(self, "revealed_indices", indices)
        object.__setattr__(self, "revealed_labels", labels)


def run_design_audit(run_id: str, group: CandidateGroup, design: AuditDesign, sidecar: TrustedSidecar, event_log: AuditEventLog, rng: np.random.Generator, *, rng_stream: str, bank_hash: str, draw_index: int, uniform_draw: float | None = None) -> AuditResult:
    design.validate(require_full_support=not design.biased and design.name != "full")
    group_hash = canonical_hash(candidate_group_payload(group))
    order_hash = canonical_hash(list(group.candidate_ids))
    draw = float(rng.random()) if uniform_draw is None else float(uniform_draw)
    if not np.isfinite(draw) or not 0 <= draw < 1:
        raise ValueError("uniform draw must be finite and in [0, 1)")
    identity = {"run": run_id, "bank": bank_hash, "problem": group.problem_id, "checkpoint": group.checkpoint_id, "group": group_hash, "candidate_order": order_hash, "draw": int(draw_index), "rng_stream": rng_stream, "uniform_draw": draw, "distribution": design.probabilities.tolist()}
    design_id = canonical_hash(identity)
    event_log.append_design(AuditDesignEvent.create(run_id, group.problem_id, group.checkpoint_id, design_id, design.probabilities, design.pi_i, design.pi_ij, rng_stream, bank_hash=bank_hash, group_hash=group_hash, candidate_order_hash=order_hash, draw_index=draw_index, uniform_draw=draw))
    subset_index = min(int(np.searchsorted(np.cumsum(design.probabilities), draw, side="right")), len(design.subsets) - 1)
    audited = design.subsets[subset_index].astype(np.int64)
    trusted = sidecar.labels(group.split_group, evaluator_token=sidecar.evaluator_token)
    revealed_indices = np.flatnonzero(audited)
    revealed_labels = trusted[revealed_indices]
    revealed = tuple(map(float, revealed_labels))
    event_log.append_outcome(AuditOutcomeEvent(run_id, group.problem_id, design_id, tuple(map(int, audited)), revealed))
    return AuditResult(audited, revealed_indices, revealed_labels, design_id)
