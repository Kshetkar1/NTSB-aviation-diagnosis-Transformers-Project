#!/usr/bin/env bash
# Run foundation reproduction tests for the current (Frozen-BN) paper path.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FROZEN_TESTS="${REPO}/Frozen-BN-Narrative-Evidence-2026-07-20/tests"
export PYTHONPATH="${REPO}/shared/code:${REPO}/Frozen-BN-Narrative-Evidence-2026-07-20/code:${PYTHONPATH:-}"

PY="${PYTHON:-python3}"
TESTS=(
  reproduce_all_examples.py
  zhang_conditional_method.py
  reproduce_fig3_from_tables.py
)

for name in "${TESTS[@]}"; do
  echo ""
  echo "=== ${name} ==="
  "${PY}" "${FROZEN_TESTS}/${name}"
done

echo ""
echo "All foundation reproduction scripts completed successfully."
