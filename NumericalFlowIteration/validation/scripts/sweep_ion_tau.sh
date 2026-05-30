#!/usr/bin/env bash
# Sweep tau for ion acoustic Landau damping.
# Runs BOTH NuFI (kinetic GPU, quasineutral Boltzmann-electron Poisson)
# and Hammett-Perkins fluid model, then plots both against the analytical rate.
#
# Edit parameters below, then run:
#   chmod +x validation/sweep_ion_tau.sh && ./validation/sweep_ion_tau.sh

set -e

# --- NuFI parameters ---
K=0.0628318530   # fixed wavenumber (= 2pi/100)
TAU_MIN=0.1
TAU_MAX=1.25
N_POINTS=16
NX=32
NU=128       # higher velocity resolution pushes recurrence to later times
DT=0.1
T_END=200.0

# --- Hammett-Perkins parameters ---
HP_MODEL="${HP_MODEL:-4moment_hammett_perkins}"
HP_NX=512
HP_T_END=450.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN_DIR="$ROOT_DIR/bin"
RESULTS_DIR="$SCRIPT_DIR/../results/$HP_MODEL"
mkdir -p "$RESULTS_DIR"
NUFI_RESULTS="$RESULTS_DIR/nufi_ion_tau_results.txt"
HP_RESULTS="$RESULTS_DIR/hp_ion_tau_results.txt"

rm -f "$NUFI_RESULTS" "$HP_RESULTS"
echo "=== Ion acoustic damping sweep: NuFI GPU + Hammett-Perkins ==="
echo "k=$K   tau: [$TAU_MIN, $TAU_MAX]   N=$N_POINTS"
echo "NuFI:  Nx=$NX  Nu=$NU  dt=$DT  T=$T_END"
echo "HP:    model=$HP_MODEL  nx=$HP_NX  T=$HP_T_END"
echo ""

TAU_VALUES=$(python3 -c "import numpy as np; print(*np.linspace($TAU_MIN, $TAU_MAX, $N_POINTS).round(6))")

for tau in $TAU_VALUES; do
    echo "--- tau=$tau ---"

    # --- NuFI run ---
    echo -n "  [NuFI]  patching config..."
    python3 "$SCRIPT_DIR/generate/gen_config_ion.py" "$tau" "$K" "$NX" "$NU" "$DT" "$T_END"
    echo -n "  compiling..."
    cd "$ROOT_DIR"
    # Must rebuild libnufi.a first (cuda_kernel.cu contains f0 compiled from config.hpp)
    make -C nufi -j$(nproc) > /dev/null 2>&1
    make -C bin test_nufi_gpu_1d -j$(nproc) > /dev/null 2>&1
    echo -n "  running..."
    cd "$BIN_DIR"
    rm -f statistics.csv E_*.txt rho_*.txt f_*.txt phase_flow_*.txt s_*.txt coeffs_*.txt
    ./test_nufi_gpu_1d --no-diagnostics > /dev/null 2>&1
    gamma_nufi=$(python3 "$SCRIPT_DIR/run/extract_gamma_nufi.py" "$BIN_DIR/statistics.csv")
    echo "$tau $gamma_nufi" >> "$NUFI_RESULTS"
    echo "  gamma=$gamma_nufi"

    # --- Hammett-Perkins run ---
    echo -n "  [HP]    running..."
    gamma_hp=$(python3 "$SCRIPT_DIR/run/run_hp_ion.py" "$tau" "$K" "$HP_NX" "$HP_T_END" "$HP_MODEL")
    echo "$tau $gamma_hp" >> "$HP_RESULTS"
    echo "  gamma=$gamma_hp"
done

echo ""
echo "Sweep done."
echo "  NuFI results: $NUFI_RESULTS"
echo "  HP results:   $HP_RESULTS"
echo "Plotting..."
cd "$ROOT_DIR"
python3 "$SCRIPT_DIR/plot/compare_ion.py" --k "$K" --models "$HP_MODEL"
