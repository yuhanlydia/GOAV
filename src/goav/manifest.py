"""Checksum-bound immutable artifact manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class ArtifactManifest:
    files: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, paths: Iterable[str | Path]) -> "ArtifactManifest":
        return cls(tuple((str(Path(path).resolve()), file_digest(path)) for path in sorted(map(Path, paths))))

    def verify(self) -> None:
        for name, expected in self.files:
            path = Path(name)
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError(f"artifact tamper detected: {name}")
