#!/usr/bin/env bash
# Sweep tau for an ion Maxwellian initial condition using NuFI.
#
# IMPORTANT PHYSICS NOTE:
#   NuFI uses Gauss's law Poisson (d²phi/dx² = delta_n_i).
#   This simulates ION PLASMA OSCILLATIONS with a fixed electron background,
#   NOT ion acoustic Landau damping. The fluid tau sweep uses quasineutral
#   Boltzmann electrons (phi = ln(n_i)), which is a different physical model.
#   Direct comparison with the fluid's tau sweep results is therefore NOT valid.
#   Ion acoustic validation requires modifying NuFI's Poisson step.
#
# Edit parameters below, then run:
#   chmod +x validation/sweep_ion_tau.sh && ./validation/sweep_ion_tau.sh

set -e

K=0.0628318530   # fixed wavenumber (same as fluid tau sweep)
TAU_MIN=0.05
TAU_MAX=1.25
N_POINTS=16
NX=128
NU=512
DT=0.05
T_END=100.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$ROOT_DIR/bin"
RESULTS="$SCRIPT_DIR/nufi_ion_tau_results.txt"

rm -f "$RESULTS"
echo "=== NuFI ion oscillation tau sweep (NOTE: Gauss Poisson, not ion acoustic) ==="
echo "k=$K   tau: [$TAU_MIN, $TAU_MAX]   N=$N_POINTS   Nx=$NX  Nu=$NU  dt=$DT  t_end=$T_END"
echo ""

TAU_VALUES=$(python3 -c "import numpy as np; print(*np.linspace($TAU_MIN, $TAU_MAX, $N_POINTS).round(6))")

for tau in $TAU_VALUES; do
    echo -n "tau=$tau  patching config..."
    python3 "$SCRIPT_DIR/gen_config_ion.py" "$tau" "$K" "$NX" "$NU" "$DT" "$T_END"

    echo -n "  compiling..."
    cd "$ROOT_DIR"
    make test_nufi_cpu_1d -j4 > /dev/null 2>&1
    echo -n "  running..."

    cd "$BIN_DIR"
    rm -f stats.txt E_*.txt
    ./test_nufi_cpu_1d > /dev/null 2>&1

    gamma=$(python3 "$SCRIPT_DIR/extract_gamma_nufi.py")
    echo "$tau $gamma" >> "$RESULTS"
    echo "  gamma=$gamma"
done

echo ""
echo "Sweep done. Results in $RESULTS"
echo "Plotting..."
cd "$ROOT_DIR"
python3 "$SCRIPT_DIR/compare_ion.py" --k "$K"
