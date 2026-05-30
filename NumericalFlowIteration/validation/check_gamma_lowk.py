#!/usr/bin/env python3
"""
Run two low-k electron Landau damping cases for both NuFI and HP,
plot E_el vs t with the gamma fit line overlaid.

Usage:
    python3 validation/check_gamma_lowk.py
"""

import os
import sys
import subprocess
import math
import numpy as np
import matplotlib.pyplot as plt

# ── parameters ──────────────────────────────────────────────────────────────
K_VALUES = [0.65, 0.58]
NX_NUFI  = 64
NU_NUFI  = 256
DT       = 0.1
T_END    = 16.0
HP_NX    = 256
HP_T_END = 60.0
HP_MODEL = "hammett_perkins"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
BIN_DIR    = os.path.join(ROOT_DIR, "bin")
PROJ_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, PROJ_DIR)


# ── NuFI helpers ─────────────────────────────────────────────────────────────
def run_nufi(k):
    subprocess.run(
        ["python3", os.path.join(SCRIPT_DIR, "gen_config_electron.py"),
         str(k), str(NX_NUFI), str(NU_NUFI), str(DT), str(T_END)],
        check=True
    )
    ncpu = os.cpu_count() or 4
    subprocess.run(["make", "-C", os.path.join(ROOT_DIR, "nufi"), f"-j{ncpu}"],
                   check=True, capture_output=True)
    subprocess.run(["make", "-C", BIN_DIR, "test_nufi_gpu_1d", f"-j{ncpu}"],
                   check=True, capture_output=True)
    for pat in ["statistics.csv", "E_*.txt", "rho_*.txt", "f_*.txt",
                "phase_flow_*.txt", "s_*.txt", "coeffs_*.txt"]:
        subprocess.run(f"rm -f {os.path.join(BIN_DIR, pat)}", shell=True)
    subprocess.run(["./test_nufi_gpu_1d"], cwd=BIN_DIR,
                   check=True, capture_output=True)

    csv = os.path.join(BIN_DIR, "statistics.csv")
    rows = []
    with open(csv) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('"'):
                continue
            parts = line.split(";")
            if len(parts) < 4:
                continue
            try:
                rows.append([float(p.strip()) for p in parts])
            except ValueError:
                continue
    return np.array(rows)


def fit_nufi(data):
    t    = data[:, 0]
    E_el = data[:, 3]
    mask = E_el > 0
    t_m, lnE = t[mask], np.log(E_el[mask])
    i_peak = np.argmax(E_el[mask])
    n = len(t_m)
    i_start = i_peak if i_peak <= n * 0.7 else max(1, n // 10)
    i_end   = n      if i_peak <= n * 0.7 else i_peak + 1
    t_fit, lE_fit = t_m[i_start:i_end], lnE[i_start:i_end]
    if len(t_fit) < 3:
        t_fit, lE_fit = t_m, lnE
    slope = np.polyfit(t_fit, lE_fit, 1)[0]
    gamma = slope / 2.0
    return t_m, E_el[mask], gamma, t_fit, np.exp(lE_fit[0]) * np.exp(slope * (t_fit - t_fit[0]))


# ── HP helpers ────────────────────────────────────────────────────────────────
def run_hp(k):
    import importlib
    import matplotlib
    matplotlib.use("Agg")

    os.chdir(PROJ_DIR)
    results_dir = os.path.join(PROJ_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    if "landau_damping" in sys.modules:
        del sys.modules["landau_damping"]
    import landau_damping as sim

    sim.k             = k
    sim.L             = 2.0 * np.pi / k
    sim.nx            = HP_NX
    sim.ionDamping    = False
    sim.t_end         = HP_T_END
    sim.MODEL         = HP_MODEL
    sim.plot_interval = int(1e9)
    sim.main()

    data    = np.load(os.path.join(PROJ_DIR, "results", "efield_data_electron.npz"),
                      allow_pickle=True)
    t       = data["t"]
    nx_data = int(data["nx"].item())
    E_kmode = data["E_kmode_real"] + 1j * data["E_kmode_imag"]
    amp     = np.abs(E_kmode) * 2.0 / nx_data
    return t, amp


def fit_hp(t, amp):
    i_peak     = np.argmax(amp)
    t_f, a_f   = t[i_peak:], amp[i_peak:]
    mask       = a_f > 0
    t_f, a_f   = t_f[mask], a_f[mask]
    if len(t_f) < 3:
        return float("nan"), None, None, None
    slope      = np.polyfit(t_f, np.log(a_f), 1)[0]
    gamma      = slope
    fit_line   = a_f[0] * np.exp(slope * (t_f - t_f[0]))
    return gamma, t_f, a_f, fit_line


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Electron Landau damping — low-k gamma extraction check", fontsize=12)

    for row, k in enumerate(K_VALUES):
        ax_nufi = axes[row, 0]
        ax_hp   = axes[row, 1]

        # ── NuFI ──
        print(f"\n=== NuFI  k={k} ===")
        nufi_data = run_nufi(k)
        t_m, E_m, gamma_n, t_fit, fit_line = fit_nufi(nufi_data)
        ax_nufi.semilogy(t_m, E_m, color="steelblue", lw=1.2, label="E_el (NuFI)")
        ax_nufi.semilogy(t_fit, fit_line, "r--", lw=2,
                         label=f"fit  γ = {gamma_n:.5f}")
        ax_nufi.set_title(f"NuFI  k={k}")
        ax_nufi.set_xlabel("t")
        ax_nufi.set_ylabel("Electric Energy")
        ax_nufi.legend(fontsize=9)
        ax_nufi.grid(which="both", ls="--", lw=0.4)
        print(f"  γ_NuFI = {gamma_n:.6f}")

        # ── HP ──
        print(f"=== HP    k={k}  model={HP_MODEL} ===")
        t_hp, amp_hp = run_hp(k)
        gamma_h, t_hf, a_hf, fit_h = fit_hp(t_hp, amp_hp)
        ax_hp.semilogy(t_hp, amp_hp, color="orange", lw=1.2, label="|E_k| (HP)")
        if t_hf is not None:
            ax_hp.semilogy(t_hf, fit_h, "r--", lw=2,
                           label=f"fit  γ = {gamma_h:.5f}")
        ax_hp.set_title(f"HP ({HP_MODEL})  k={k}")
        ax_hp.set_xlabel("t")
        ax_hp.set_ylabel("|E_k|")
        ax_hp.legend(fontsize=9)
        ax_hp.grid(which="both", ls="--", lw=0.4)
        print(f"  γ_HP   = {gamma_h:.6f}")

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "results", "check_gamma_lowk.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=180)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    main()
