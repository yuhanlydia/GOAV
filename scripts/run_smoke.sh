#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_root}/artifacts/smoke}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${GOAV_PYTHON:-}"
if [[ -z "${python_bin}" && -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
elif [[ -z "${python_bin}" ]]; then
  python_bin="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${python_bin}" ]]; then
  echo "python3 or python is required" >&2
  exit 127
fi

"${python_bin}" -m goav doctor
"${python_bin}" -m goav prepare "${repo_root}/configs/experiments/smoke.yaml" --output "${output}"
"${python_bin}" -m goav run "${repo_root}/configs/experiments/smoke.yaml" --output "${output}"
"${python_bin}" -m goav verify "${output}"
