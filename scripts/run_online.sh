#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 5 ]]; then
  echo "usage: $0 CONFIG BATCH_NPZ BANK_HASH ALGORITHM OUTPUT" >&2
  exit 2
fi
config="$1"
batch_npz="$2"
bank_hash="$3"
algorithm="$4"
output="$5"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m goav doctor
python -m goav online-update "${config}" --batch-npz "${batch_npz}" --bank-hash "${bank_hash}" --algorithm "${algorithm}" --output "${output}"
python -m goav verify "${output}"
