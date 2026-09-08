#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_root}/artifacts/smoke}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m goav doctor
python -m goav prepare "${repo_root}/configs/experiments/smoke.yaml" --output "${output}"
python -m goav run "${repo_root}/configs/experiments/smoke.yaml" --output "${output}"
python -m goav verify "${output}"
