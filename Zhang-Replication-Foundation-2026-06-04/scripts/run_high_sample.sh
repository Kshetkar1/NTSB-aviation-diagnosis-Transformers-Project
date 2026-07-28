#!/usr/bin/env bash
# Runs the full Zhang replication suite at 99,999,999 samples (matching his
# XDSL declaration) and writes outputs with a `_99M` suffix so the existing
# 50K-sample outputs remain intact for side-by-side comparison.
#
# Estimated wall-clock: ~7-8 hours.
#
# Usage:
#   bash Zhang_Replication_Runner/run_high_sample.sh
#
# Live progress is appended to outputs/run_99M.log
#
# Resume-friendly: if you kill the run and re-launch, scripts that already
# wrote their *_99M.json output will be skipped.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/Users/kanushetkar/opt/anaconda3/envs/zhang2021/bin/python"
SAMPLES=99999999
SEED=42
SUFFIX="_99M"
LOG="outputs/run${SUFFIX}.log"

mkdir -p outputs
touch "$LOG"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

run_step() {
    local script="$1"
    local out_stem="$2"
    if [[ -f "outputs/${out_stem}${SUFFIX}.json" ]]; then
        echo "[$(ts)] SKIP ${script} -- ${out_stem}${SUFFIX}.json already exists" | tee -a "$LOG"
        return 0
    fi
    echo "[$(ts)] BEGIN ${script}" | tee -a "$LOG"
    local t0=$(date +%s)
    "$PYTHON" -u "$script" --samples "$SAMPLES" --seed "$SEED" --out-suffix "$SUFFIX" 2>&1 | tee -a "$LOG"
    local t1=$(date +%s)
    echo "[$(ts)] END ${script} (elapsed $((t1 - t0)) s)" | tee -a "$LOG"
}

echo "=========================================" | tee -a "$LOG"
echo "[$(ts)] LAUNCH high-sample replication" | tee -a "$LOG"
echo "  samples=${SAMPLES}  seed=${SEED}  suffix=${SUFFIX}" | tee -a "$LOG"
echo "=========================================" | tee -a "$LOG"

run_step "01_table8_sensitivity.py"  "table8_sensitivity"
run_step "02_fig11_multi_evidence.py" "fig11_multi_evidence"
run_step "03_fig12_pilot_error.py"   "fig12_pilot_error"
run_step "04_table9_engine_power.py" "table9_engine_power"

echo "[$(ts)] running validation report" | tee -a "$LOG"
"$PYTHON" -u 05_validation_report.py --in-suffix "$SUFFIX" --out-suffix "$SUFFIX" 2>&1 | tee -a "$LOG"

echo "[$(ts)] DONE -- 99M artifacts in outputs/ with suffix ${SUFFIX}" | tee -a "$LOG"

# Chain into the multi-seed band run (~25 min) for the strongest defensible
# comparison story. See run_seed_band.sh for details.
# Note: cwd is already $(dirname "$0") (set at top of script), so use ./.
echo "[$(ts)] chaining into multi-seed band run" | tee -a "$LOG"
bash ./run_seed_band.sh
echo "[$(ts)] FULL PIPELINE COMPLETE -- 99M validation + 5-seed noise band" | tee -a "$LOG"
