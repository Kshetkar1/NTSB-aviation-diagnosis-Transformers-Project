#!/usr/bin/env bash
# Shows progress of the 99M-sample replication run + the seed-band watcher.
# Re-runs every 30 seconds. Quit with Ctrl-C.
#
# Usage:
#   bash Zhang_Replication_Runner/status.sh
#
# Or for a one-shot snapshot:
#   bash Zhang_Replication_Runner/status.sh --once

set -euo pipefail

cd "$(dirname "$0")"

LOG="outputs/run_99M.log"
SEEDLOG="outputs/run_seed_band.log"

# Inference counts per script (used to compute % progress within the 99M run)
inferences_for() {
    case "$1" in
        01_table8_sensitivity.py) echo 12 ;;
        02_fig11_multi_evidence.py) echo 5 ;;
        03_fig12_pilot_error.py) echo 3 ;;
        04_table9_engine_power.py) echo 5 ;;
        *) echo 0 ;;
    esac
}

TOTAL_INFERENCES=25  # sum of above
SECS_PER_INFERENCE=1050  # ~17.5 minutes at 99M samples (from benchmark)

snapshot() {
    clear || true
    echo "================================================================"
    echo "  Zhang 99M-sample replication -- progress  ($(date "+%Y-%m-%d %H:%M:%S"))"
    echo "================================================================"
    echo ""

    # PIDs running
    local pid_99m
    pid_99m=$(pgrep -f "run_high_sample.sh" 2>/dev/null | head -n1 || true)
    local pid_watcher
    pid_watcher=$(pgrep -f "_watch_then_band.sh" 2>/dev/null | head -n1 || true)

    if [[ -n "$pid_99m" ]]; then
        echo "  99M run process: PID $pid_99m  [RUNNING]"
    elif [[ -f "outputs/validation_report_99M.xlsx" ]]; then
        echo "  99M run process: COMPLETE (validation_report_99M.xlsx exists)"
    else
        echo "  99M run process: NOT RUNNING (and not yet complete)"
    fi

    if [[ -n "$pid_watcher" ]]; then
        echo "  band watcher:    PID $pid_watcher  [waiting for 99M to finish]"
    elif [[ -f "outputs/seed_band_report.xlsx" ]]; then
        echo "  band watcher:    COMPLETE (seed_band_report.xlsx exists)"
    else
        echo "  band watcher:    NOT RUNNING"
    fi

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Per-script progress (99M run)"
    echo "----------------------------------------------------------------"

    if [[ ! -f "$LOG" ]]; then
        echo "  (no log file yet)"
    else
        local total_completed=0
        for script in "01_table8_sensitivity.py" "02_fig11_multi_evidence.py" \
                      "03_fig12_pilot_error.py" "04_table9_engine_power.py"; do
            local total
            total=$(inferences_for "$script")
            local stem
            stem="$(echo "$script" | sed 's/\.py$//' | sed 's/^[0-9]*_//')"
            local out_json="outputs/${stem}_99M.json"

            local state="pending"
            local done_count=0
            if grep -q "BEGIN ${script}" "$LOG" 2>/dev/null; then
                state="running"
                # Progress within a script: count printed result lines using awk
                # (avoids grep -c quirk where exit-1-on-no-matches forces fallback).
                case "$script" in
                    01_*) done_count=$(awk '/P\(strut\)=/{c++} END{print c+0}' "$LOG") ;;
                    02_*) done_count=$(awk '/step [0-9]+/{c++} END{print c+0}' "$LOG") ;;
                    03_*) done_count=$(awk '/phase [a-z]+:/{c++} END{print c+0}' "$LOG") ;;
                    04_*) done_count=$(awk '/evidence set:/{c++} END{print c+0}' "$LOG") ;;
                esac
                done_count=${done_count:-0}
                if (( done_count > total )); then done_count=$total; fi
            fi
            if [[ -f "$out_json" ]]; then
                state="DONE"
                done_count=$total
            fi

            total_completed=$((total_completed + done_count))
            # Draw simple progress bar without relying on seq edge cases
            local bar=""
            local i=0
            while [[ $i -lt $total ]]; do
                if [[ $i -lt $done_count ]]; then
                    bar="${bar}#"
                else
                    bar="${bar}."
                fi
                i=$((i + 1))
            done
            printf "  %-30s [%-12s] %2d/%2d  %s\n" \
                "$script" "$bar" "$done_count" "$total" "$state"
        done

        echo ""
        local pct=$(( total_completed * 100 / TOTAL_INFERENCES ))
        local remain=$(( TOTAL_INFERENCES - total_completed ))
        local secs_remain=$(( remain * SECS_PER_INFERENCE ))
        local hrs=$(( secs_remain / 3600 ))
        local mins=$(( (secs_remain % 3600) / 60 ))
        echo "  TOTAL:  ${total_completed}/${TOTAL_INFERENCES} inferences (${pct}%)   ETA: ~${hrs}h ${mins}m"
    fi

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Last 8 lines of run_99M.log"
    echo "----------------------------------------------------------------"
    if [[ -f "$LOG" ]]; then
        tail -n 8 "$LOG" | sed 's/^/  /'
    else
        echo "  (no log file yet)"
    fi

    if [[ -f "$SEEDLOG" ]]; then
        echo ""
        echo "----------------------------------------------------------------"
        echo "  Last 4 lines of run_seed_band.log"
        echo "----------------------------------------------------------------"
        tail -n 4 "$SEEDLOG" | sed 's/^/  /'
    fi

    echo ""
    echo "  refreshes every 30s -- Ctrl-C to quit (does NOT kill the run)"
}

if [[ "${1:-}" == "--once" ]]; then
    snapshot
else
    while true; do
        snapshot
        sleep 30
    done
fi
