"""Candidate/test banks and the trusted-label type firewall."""

from __future__ import annotations

import json
import io
import secrets
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import canonical_hash

EXPECTED_TEST_COMPOSITION = {"curated": 12, "generated": 12, "fault_targeted": 12, "metamorphic": 8, "control": 4}


@dataclass(frozen=True)
class CheapTest:
    test_id: str
    kind: str
    content: str


@dataclass(frozen=True)
class CandidateGroup:
    problem_id: str
    checkpoint_id: str
    candidate_ids: tuple[str, ...]
    tests: tuple[CheapTest, ...]
    cheap_outcomes: np.ndarray
    prompt: str = ""
    candidate_texts: tuple[str, ...] = ()
    duplicate_cluster_id: str | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.cheap_outcomes, dtype=np.int8)
        if len(self.candidate_ids) != 8:
            raise ValueError("candidate groups require K=8")
        if matrix.shape != (8, len(self.tests)):
            raise ValueError("cheap outcome shape must be K by tests")
        if self.candidate_texts and len(self.candidate_texts) != len(self.candidate_ids):
            raise ValueError("candidate text must align with candidate IDs")
        object.__setattr__(self, "cheap_outcomes", matrix.copy())
        if self.test_composition != EXPECTED_TEST_COMPOSITION:
            raise ValueError(f"test composition must be {EXPECTED_TEST_COMPOSITION}")

    @property
    def split_group(self) -> tuple[str, str]:
        return self.problem_id, self.checkpoint_id

    @property
    def base_split_id(self) -> str:
        return self.duplicate_cluster_id or self.problem_id

    @property
    def test_composition(self) -> dict[str, int]:
        return {kind: sum(test.kind == kind for test in self.tests) for kind in EXPECTED_TEST_COMPOSITION}


@dataclass(frozen=True)
class TrainerCandidateView:
    problem_id: str
    checkpoint_id: str
    candidate_ids: tuple[str, ...]
    tests: tuple[CheapTest, ...]
    cheap_outcomes: np.ndarray
    prompt: str
    candidate_texts: tuple[str, ...]
    duplicate_cluster_id: str | None


def trainer_view(group: CandidateGroup) -> TrainerCandidateView:
    matrix = group.cheap_outcomes.copy()
    matrix.flags.writeable = False
    return TrainerCandidateView(group.problem_id, group.checkpoint_id, group.candidate_ids, group.tests, matrix, group.prompt, group.candidate_texts, group.duplicate_cluster_id)


def candidate_group_payload(group: CandidateGroup) -> dict:
    return {
        "problem_id": group.problem_id, "checkpoint_id": group.checkpoint_id,
        "duplicate_cluster_id": group.duplicate_cluster_id, "prompt": group.prompt,
        "candidate_ids": list(group.candidate_ids), "candidate_texts": list(group.candidate_texts),
        "tests": [{"test_id": test.test_id, "kind": test.kind, "content": test.content} for test in group.tests],
        "cheap_outcomes": group.cheap_outcomes.tolist(),
    }


def bank_hash(groups: Sequence[CandidateGroup]) -> str:
    return canonical_hash([candidate_group_payload(group) for group in sorted(groups, key=lambda item: (item.base_split_id, item.problem_id, item.checkpoint_id))])


class TrustedSidecar:
    """Evaluator-owned labels; access requires an unforgeable run-local token."""

    def __init__(self, labels: Mapping[tuple[str, str], np.ndarray], token: str):
        self._labels = {key: np.asarray(value, dtype=float).copy() for key, value in labels.items()}
        self._token = token

    @classmethod
    def create(cls, labels: Mapping[tuple[str, str], np.ndarray]) -> "TrustedSidecar":
        return cls(labels, secrets.token_hex(32))

    @property
    def evaluator_token(self) -> str:
        return self._token

    def labels(self, split_group: tuple[str, str] | object, *, evaluator_token: str) -> np.ndarray:
        if evaluator_token != self._token or not isinstance(split_group, tuple):
            raise PermissionError("trusted labels are evaluator-only")
        result = self._labels[split_group].copy()
        result.flags.writeable = False
        return result


def write_bank(jsonl_path: str | Path, npz_path: str | Path, groups: Sequence[CandidateGroup]) -> None:
    arrays: dict[str, np.ndarray] = {}
    lines = []
    for index, group in enumerate(sorted(groups, key=lambda item: item.split_group)):
        key = f"group_{index}"
        arrays[key] = group.cheap_outcomes
        payload = {
            "candidate_ids": list(group.candidate_ids), "checkpoint_id": group.checkpoint_id,
            "matrix_key": key, "problem_id": group.problem_id,
            "prompt": group.prompt, "candidate_texts": list(group.candidate_texts),
            "duplicate_cluster_id": group.duplicate_cluster_id,
            "tests": [{"content": test.content, "kind": test.kind, "test_id": test.test_id} for test in group.tests],
        }
        lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    Path(jsonl_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def read_bank(jsonl_path: str | Path, npz_path: str | Path) -> list[CandidateGroup]:
    with np.load(npz_path, allow_pickle=False) as arrays:
        result = []
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            tests = tuple(CheapTest(**test) for test in payload["tests"])
            result.append(CandidateGroup(payload["problem_id"], payload["checkpoint_id"], tuple(payload["candidate_ids"]), tests, arrays[payload["matrix_key"]], prompt=payload.get("prompt", ""), candidate_texts=tuple(payload.get("candidate_texts", ())), duplicate_cluster_id=payload.get("duplicate_cluster_id")))
    return result


def read_trusted_sidecar(path: str | Path, *, expected_bank_hash: str) -> TrustedSidecar:
    """Load evaluator-owned labels and require an exact prepared-bank binding."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(payload) is not dict or set(payload) != {"bank_hash", "labels"}:
        raise ValueError("trusted sidecar must contain exactly bank_hash and labels")
    if payload["bank_hash"] != expected_bank_hash:
        raise ValueError("trusted sidecar bank hash mismatch")
    rows = payload["labels"]
    if type(rows) is not list:
        raise ValueError("trusted sidecar labels must be a list")
    labels: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"problem_id", "checkpoint_id", "values"}:
            raise ValueError("trusted sidecar label row has invalid schema")
        key = (row["problem_id"], row["checkpoint_id"])
        values = np.asarray(row["values"], dtype=float)
        if key in labels or values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("trusted sidecar labels must be unique finite K=8 vectors")
        labels[key] = values
    return TrustedSidecar.create(labels)
