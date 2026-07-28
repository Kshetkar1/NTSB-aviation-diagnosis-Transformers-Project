#!/usr/bin/env bash
# Multi-seed band run: executes scripts 01-04 at 1M samples for 5 seeds
# (1, 7, 42, 100, 9999) so we can characterize the noise band Zhang's single
# unseeded run sat within.
#
# Total wall-clock: ~25 minutes (4 scripts × ~25 inferences total × 5 seeds × ~10s/inference)
# Outputs: outputs/{stem}_seed{N}.json
#          outputs/seed_band_report.xlsx
#          outputs/SEED_BAND.md

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="/Users/kanushetkar/opt/anaconda3/envs/zhang2021/bin/python"
SAMPLES=1000000
SEEDS=(1 7 42 100 9999)
LOG="outputs/run_seed_band.log"

mkdir -p outputs
touch "$LOG"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

run_step() {
    local script="$1"
    local out_stem="$2"
    local seed="$3"
    local suffix="_seed${seed}"
    if [[ -f "outputs/${out_stem}${suffix}.json" ]]; then
        echo "[$(ts)] SKIP ${script} seed=${seed} -- exists" | tee -a "$LOG"
        return 0
    fi
    echo "[$(ts)] BEGIN ${script} seed=${seed}" | tee -a "$LOG"
    local t0=$(date +%s)
    "$PYTHON" -u "$script" --samples "$SAMPLES" --seed "$seed" --out-suffix "$suffix" 2>&1 | tee -a "$LOG"
    local t1=$(date +%s)
    echo "[$(ts)] END ${script} seed=${seed} (elapsed $((t1 - t0)) s)" | tee -a "$LOG"
}

echo "=========================================" | tee -a "$LOG"
echo "[$(ts)] LAUNCH multi-seed band run" | tee -a "$LOG"
echo "  samples=${SAMPLES}  seeds=${SEEDS[*]}" | tee -a "$LOG"
echo "=========================================" | tee -a "$LOG"

for seed in "${SEEDS[@]}"; do
    run_step "01_table8_sensitivity.py"  "table8_sensitivity"  "$seed"
    run_step "02_fig11_multi_evidence.py" "fig11_multi_evidence" "$seed"
    run_step "03_fig12_pilot_error.py"   "fig12_pilot_error"   "$seed"
    run_step "04_table9_engine_power.py" "table9_engine_power" "$seed"
done

echo "[$(ts)] running seed-band aggregator" | tee -a "$LOG"
"$PYTHON" -u 07_seed_band.py 2>&1 | tee -a "$LOG"

echo "[$(ts)] DONE -- artifacts: outputs/seed_band_report.xlsx and outputs/SEED_BAND.md" | tee -a "$LOG"
