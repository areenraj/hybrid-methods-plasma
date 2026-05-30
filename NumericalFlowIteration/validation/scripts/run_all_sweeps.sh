#!/usr/bin/env bash
# Run ion + electron sweeps for both HP models (4 total).
# Logs each run to a timestamped file in validation/logs/.
#
# Usage:
#   chmod +x validation/run_all_sweeps.sh && ./validation/run_all_sweeps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MODELS=("hammett_perkins" "4moment_hammett_perkins")

run_sweep() {
    local script="$1"
    local model="$2"
    local label
    label="$(basename "$script" .sh)__${model}"
    local logfile="$LOG_DIR/${TIMESTAMP}_${label}.log"

    echo "======================================================"
    echo "  SWEEP : $label"
    echo "  LOG   : $logfile"
    echo "======================================================"
    HP_MODEL="$model" bash "$script" 2>&1 | tee "$logfile"
    echo ""
}

for model in "${MODELS[@]}"; do
    run_sweep "$SCRIPT_DIR/sweep_ion_tau.sh"    "$model"
    run_sweep "$SCRIPT_DIR/sweep_electron_k.sh" "$model"
done

echo "======================================================"
echo "All sweeps complete."
echo "Results in: $SCRIPT_DIR/../results/"
echo "Logs in:    $LOG_DIR/"
echo "======================================================"
