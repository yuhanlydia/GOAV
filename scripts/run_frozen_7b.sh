#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 5 ]]; then
  echo "usage: $0 CONFIG BANK_JSONL BANK_NPZ TRUSTED_SIDECAR OUTPUT" >&2
  exit 2
fi
config="$1"
bank_jsonl="$2"
bank_npz="$3"
trusted_sidecar="$4"
output="$5"
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
"${python_bin}" -m goav prepare "${config}" --output "${output}"
"${python_bin}" -m goav run "${config}" --output "${output}" --bank-jsonl "${bank_jsonl}" --bank-npz "${bank_npz}" --trusted-sidecar "${trusted_sidecar}" --backend transformers
"${python_bin}" -m goav verify "${output}"
