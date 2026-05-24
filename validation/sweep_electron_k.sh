#!/usr/bin/env bash
# Sweep k for electron Landau damping using NuFI (kinetic Vlasov-Poisson).
# For each k: patches config.hpp, rebuilds test_nufi_cpu_1d, runs it, extracts gamma.
#
# Edit the parameters below, then run:
#   chmod +x validation/sweep_electron_k.sh && ./validation/sweep_electron_k.sh

set -e

K_MIN=0.2
K_MAX=0.65
N_POINTS=20
NX=64        # grid points (64 per wavelength since domain = one wavelength)
NU=512       # velocity quadrature points
DT=0.05
T_END=60.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$ROOT_DIR/bin"
RESULTS="$SCRIPT_DIR/nufi_electron_k_results.txt"

rm -f "$RESULTS"
echo "=== NuFI electron Landau damping k sweep ==="
echo "k: [$K_MIN, $K_MAX]   N=$N_POINTS   Nx=$NX  Nu=$NU  dt=$DT  t_end=$T_END"
echo ""

K_VALUES=$(python3 -c "import numpy as np; print(*np.linspace($K_MIN, $K_MAX, $N_POINTS).round(6))")

for k_val in $K_VALUES; do
    echo -n "k=$k_val  patching config..."
    python3 "$SCRIPT_DIR/gen_config_electron.py" "$k_val" "$NX" "$NU" "$DT" "$T_END"

    echo -n "  compiling..."
    cd "$ROOT_DIR"
    make test_nufi_cpu_1d -j4 > /dev/null 2>&1
    echo -n "  running..."

    cd "$BIN_DIR"
    rm -f stats.txt E_*.txt  # clean previous outputs
    ./test_nufi_cpu_1d > /dev/null 2>&1

    gamma=$(python3 "$SCRIPT_DIR/extract_gamma_nufi.py")
    echo "$k_val $gamma" >> "$RESULTS"
    echo "  gamma=$gamma"
done

echo ""
echo "Sweep done. Results in $RESULTS"
echo "Plotting..."
cd "$ROOT_DIR"
python3 "$SCRIPT_DIR/compare_electron.py"
