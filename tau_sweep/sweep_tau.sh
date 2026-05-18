#!/usr/bin/env bash
# Sweep tau at fixed k, extract gamma, plot
# Edit the parameters below, then run:
#   chmod +x tau_sweep/sweep_tau.sh && ./tau_sweep/sweep_tau.sh

set -e

K=0.0628318531
TAU_MIN=0.05
TAU_MAX=1.25
N_POINTS=16

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS="$PARENT_DIR/results/sweep_results.txt"

cd "$PARENT_DIR"
rm -f "$RESULTS"

echo "=== Ion acoustic damping sweep ==="
echo "k=$K   tau: [$TAU_MIN, $TAU_MAX]   N=$N_POINTS"
echo ""

TAU_VALUES=$(python3 -c "import numpy as np; print(*np.linspace($TAU_MIN, $TAU_MAX, $N_POINTS).round(6))")

for tau in $TAU_VALUES; do
    echo -n "tau=$tau  running sim..."

    MPLBACKEND=Agg python3 -c "
import matplotlib
matplotlib.use('Agg')
import Finite_Volume as sim
sim.tau = $tau
sim.ionDamping = True
sim.plot_interval = 10**9
sim.main()
"

    gamma=$(MPLBACKEND=Agg python3 "$SCRIPT_DIR/extract_gamma.py")
    echo "$tau $gamma" >> "$RESULTS"
    echo "  gamma=$gamma"
done

echo ""
echo "Sweep done. Results in $RESULTS"
echo "Plotting..."
python3 "$SCRIPT_DIR/ion_tau_plot.py" --k "$K"
