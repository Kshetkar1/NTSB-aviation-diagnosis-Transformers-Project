#!/usr/bin/env bash
# Launch apps-parity Streamlit demo (does not touch ../apps/).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# Paper-default retrieval index: Zhang build window 1982-2006 (when files exist).
unset NTSB_FULL_CORPUS NTSB_USE_TRAIN_INDEX

export PYTHONPATH="${ROOT}/shared/code:${ROOT}/Frozen-BN-Narrative-Evidence-2026-07-20/code:${PYTHONPATH:-}"

cd "$APP_DIR"
exec streamlit run streamlit_app.py "$@"
