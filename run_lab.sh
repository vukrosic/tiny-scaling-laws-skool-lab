#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
study="${1:-capacity}"
shift || true

printf '%s\n' '[1/3] Preparing the Python environment...'
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -u setup_dependencies.py
printf '%s\n' '[2/3] Running the CPU scaling experiment...'

case "$study" in
  capacity) .venv/bin/python -u scaling_lab.py "$@" ;;
  budget) .venv/bin/python -u skool_studies.py budget "$@" ;;
  data) .venv/bin/python -u skool_studies.py data "$@" ;;
  *) printf 'Unknown study: %s\n' "$study" >&2; exit 2 ;;
esac

printf '%s\n' '[3/3] Complete.'
