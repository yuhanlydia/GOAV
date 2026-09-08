"""Immutable JSON run artifacts with checksum verification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .manifest import ArtifactManifest
from .manifest import file_digest

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _safe_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError("artifact name must be a safe relative path")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("artifact name must be a safe relative path")
    return resolved


def _valid_binding(value: str, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a valid sha256 digest")
    return value


class ArtifactStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        if (self.directory / "checksums.json").exists():
            raise PermissionError("artifact directory is sealed")
        path = _safe_path(self.directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
        except FileExistsError:
            raise FileExistsError(f"immutable artifact already exists: {path}") from None
        return path

    def manifest(self) -> ArtifactManifest:
        return ArtifactManifest.create(path for path in self.directory.iterdir() if path.is_file())


def write_checksum_index(directory: str | Path, *, config_hash: str, bank_hash: str) -> Path:
    root = Path(directory).resolve()
    _valid_binding(config_hash, "config hash")
    _valid_binding(bank_hash, "bank hash")
    index = root / "checksums.json"
    if index.exists():
        raise FileExistsError("artifact directory is already sealed")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != index)
    payload = {
        "binding": {"config_hash": config_hash, "bank_hash": bank_hash},
        "files": {str(path.relative_to(root)): file_digest(path) for path in files},
    }
    with index.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return index


def verify_checksum_index(directory: str | Path, *, config_hash: str | None = None, bank_hash: str | None = None) -> None:
    root = Path(directory).resolve()
    index = root / "checksums.json"
    if not index.is_file():
        raise ValueError("checksums.json is missing")
    payload = json.loads(index.read_text(encoding="utf-8"))
    if set(payload) != {"binding", "files"} or type(payload["binding"]) is not dict or type(payload["files"]) is not dict:
        raise ValueError("invalid checksum index schema")
    binding = payload["binding"]
    stored_config = _valid_binding(binding.get("config_hash"), "stored config hash")
    stored_bank = _valid_binding(binding.get("bank_hash"), "stored bank hash")
    if config_hash is not None and _valid_binding(config_hash, "config hash") != stored_config:
        raise ValueError("artifact config hash binding mismatch")
    if bank_hash is not None and _valid_binding(bank_hash, "bank hash") != stored_bank:
        raise ValueError("artifact bank hash binding mismatch")
    expected = payload["files"]
    actual_names = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path != index}
    if actual_names != set(expected):
        raise ValueError("artifact file set differs from immutable checksum index")
    for name, digest in expected.items():
        path = _safe_path(root, name)
        _valid_binding(digest, f"digest for {name}")
        if file_digest(path) != digest:
            raise ValueError(f"artifact tamper detected: {name}")
