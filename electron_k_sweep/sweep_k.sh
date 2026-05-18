#!/usr/bin/env bash
# Sweep k at fixed tau=1 for electron damping, extract gamma, plot
# Edit the parameters below, then run:
#   chmod +x electron_k_sweep/sweep_k.sh && ./electron_k_sweep/sweep_k.sh

set -e

K_MIN=0.2
K_MAX=0.65
N_POINTS=20
BASE_NX=128    # base grid resolution (scaled internally per k)
T_END=60.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS="$PARENT_DIR/results/electron_k_results.txt"

cd "$PARENT_DIR"
rm -f "$RESULTS"

echo "=== Electron Landau damping sweep ==="
echo "k: [$K_MIN, $K_MAX]   N=$N_POINTS"
echo ""

K_VALUES=$(python3 -c "import numpy as np; print(*np.linspace($K_MIN, $K_MAX, $N_POINTS).round(6))")

for k_val in $K_VALUES; do
    echo -n "k=$k_val  running sim..."

    MPLBACKEND=Agg python3 -c "
import matplotlib; matplotlib.use('Agg')
import numpy as np
import Finite_Volume as sim

k_val = $k_val
sim.k          = k_val
sim.L          = 2 * np.pi / k_val
sim.ionDamping = False
sim.t_end      = $T_END
sim.plot_interval = 10**9

sim.nx         = int(np.ceil($BASE_NX * 0.5 / k_val))
sim.dx         = sim.L / sim.nx
sim.x          = np.linspace(0.0, sim.L, sim.nx, endpoint=False)
sim.k_rfft     = 2.0 * np.pi * np.fft.rfftfreq(sim.nx, d=sim.dx)
sim.abs_k_rfft = np.abs(sim.k_rfft)
sim.abs_k_rfft[0] = 1.0

sim.main()
"

    gamma=$(MPLBACKEND=Agg python3 "$SCRIPT_DIR/extract_gamma.py")
    echo "$k_val $gamma" >> "$RESULTS"
    echo "  gamma=$gamma"
done

echo ""
echo "Sweep done. Results in $RESULTS"
echo "Plotting..."
python3 "$SCRIPT_DIR/electron_k_plot.py"
