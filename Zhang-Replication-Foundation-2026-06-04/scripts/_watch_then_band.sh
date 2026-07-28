#!/usr/bin/env bash
# Polls for the 99M validation report to appear, then launches the seed-band
# run. Used because the in-flight 99M run was launched before run_high_sample.sh
# was updated to chain automatically.

set -euo pipefail

cd "$(dirname "$0")"

MARKER="outputs/validation_report_99M.xlsx"
LOG="outputs/run_seed_band.log"
mkdir -p outputs

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] watcher started -- polling for ${MARKER} every 60s" | tee -a "$LOG"

while [[ ! -f "$MARKER" ]]; do
    sleep 60
done

echo "[$(ts)] ${MARKER} appeared -- launching multi-seed band run" | tee -a "$LOG"
bash ./run_seed_band.sh
