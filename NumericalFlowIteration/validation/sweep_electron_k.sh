#!/usr/bin/env bash
# Sweep k for electron Landau damping.
# Runs BOTH NuFI (kinetic GPU) and Hammett-Perkins fluid model,
# then plots both against the analytical Landau rate.
#
# Edit the parameters below, then run:
#   chmod +x validation/sweep_electron_k.sh && ./validation/sweep_electron_k.sh

set -e

# --- NuFI parameters ---
K_MIN=0.23
K_MAX=0.65
N_POINTS=20
NX=64        # grid points in x (one wavelength domain)
NU=256       # velocity quadrature points
DT=0.1
T_END=16.0

# --- Hammett-Perkins parameters ---
HP_MODEL="4moment_hammett_perkins"   # or "4moment_hammett_perkins"
HP_NX=256                    # fluid grid points
HP_T_END=60.0                # fluid needs longer to show clean decay

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$ROOT_DIR/bin"
NUFI_RESULTS="$SCRIPT_DIR/nufi_electron_k_results.txt"
HP_RESULTS="$SCRIPT_DIR/nufi_hp_electron_k_results.txt"

rm -f "$NUFI_RESULTS" "$HP_RESULTS"
echo "=== Electron Landau damping sweep: NuFI GPU + Hammett-Perkins ==="
echo "k: [$K_MIN, $K_MAX]   N=$N_POINTS"
echo "NuFI:  Nx=$NX  Nu=$NU  dt=$DT  T=$T_END"
echo "HP:    model=$HP_MODEL  nx=$HP_NX  T=$HP_T_END"
echo ""

K_VALUES=$(python3 -c "import numpy as np; print(*np.linspace($K_MIN, $K_MAX, $N_POINTS).round(6))")

for k_val in $K_VALUES; do
    echo "--- k=$k_val ---"

    # --- NuFI run ---
    echo -n "  [NuFI]  patching config..."
    python3 "$SCRIPT_DIR/gen_config_electron.py" "$k_val" "$NX" "$NU" "$DT" "$T_END"
    echo -n "  compiling..."
    cd "$ROOT_DIR"
    make -C bin test_nufi_gpu_1d -j$(nproc) > /dev/null 2>&1
    echo -n "  running..."
    cd "$BIN_DIR"
    rm -f statistics.csv E_*.txt rho_*.txt f_*.txt phase_flow_*.txt s_*.txt coeffs_*.txt
    ./test_nufi_gpu_1d > /dev/null 2>&1
    gamma_nufi=$(python3 "$SCRIPT_DIR/extract_gamma_nufi.py" "$BIN_DIR/statistics.csv")
    echo "$k_val $gamma_nufi" >> "$NUFI_RESULTS"
    echo "  gamma=$gamma_nufi"

    # --- Hammett-Perkins run ---
    echo -n "  [HP]    running..."
    gamma_hp=$(python3 "$SCRIPT_DIR/run_hp_electron.py" "$k_val" "$HP_NX" "$HP_T_END" "$HP_MODEL")
    echo "$k_val $gamma_hp" >> "$HP_RESULTS"
    echo "  gamma=$gamma_hp"
done

echo ""
echo "Sweep done."
echo "  NuFI results: $NUFI_RESULTS"
echo "  HP results:   $HP_RESULTS"
echo "Plotting..."
cd "$ROOT_DIR"
python3 "$SCRIPT_DIR/compare_electron.py"
