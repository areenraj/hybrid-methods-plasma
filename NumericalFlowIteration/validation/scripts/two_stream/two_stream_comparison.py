#!/usr/bin/env python3
"""
Manually extract linear growth rates from two-stream simulation results.

Specify separate time windows for each solver; the script fits log(|E_k|) vs t
inside each window and reports gamma = d ln|E_k| / dt.

Usage examples
--------------
# Both solvers, same window:
python3 two_stream_comparison.py --hp-window 5 20 --nufi-window 5 20

# Different windows (e.g. HP runs hotter and saturates earlier):
python3 two_stream_comparison.py --hp-window 3 18 --nufi-window 5 22

# Custom file paths:
python3 two_stream_comparison.py --hp-window 5 20 --nufi-window 5 20 \
    --hp-file /path/to/efield_data_twostream.npz \
    --nufi-file /path/to/statistics.csv

Data sources (defaults)
-----------------------
HP  : validation/results/4moment_hammett_perkins/two_stream/efield_data_twostream.npz
NuFI: validation/results/4moment_hammett_perkins/two_stream/statistics.csv
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(
    _SCRIPT_DIR, "..", "..", "results", "4moment_hammett_perkins", "two_stream"
))


# ============================================================
# LOADERS
# ============================================================

def load_hp(path):
    d = np.load(path)
    t = d["t"]
    nx = int(d["nx"][0])
    E_kmode = np.abs(d["E_kmode_real"] + 1j * d["E_kmode_imag"]) * 2.0 / nx
    return t, E_kmode


def load_nufi(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('"'):
                continue
            parts = line.split(";")
            if len(parts) < 8:
                continue
            try:
                rows.append([float(p.strip()) for p in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No numeric data in {path}")
    data = np.array(rows)
    return data[:, 0], data[:, 7]


# ============================================================
# FIT
# ============================================================

def fit_window(t, amp, t_start, t_end):
    mask = (t >= t_start) & (t <= t_end) & (amp > 0)
    n = mask.sum()
    if n < 3:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(t[mask], np.log(amp[mask]), 1)
    log_amp    = np.log(amp[mask])
    log_fitted = slope * t[mask] + intercept
    ss_res = np.sum((log_amp - log_fitted) ** 2)
    ss_tot = np.sum((log_amp - log_amp.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, np.exp(intercept), r2


# ============================================================
# PLOT
# ============================================================

def make_plot(t_hp, E_hp, t_nufi, E_nufi,
              hp_window, nufi_window,
              gamma_hp, A_hp, r2_hp,
              gamma_nufi, A_nufi, r2_nufi,
              outdir="."):

    fig, ax = plt.subplots(figsize=(9, 5))

    GREEN  = "#6aA84F"
    PURPLE = "#9B4E8F"

    ax.semilogy(t_hp,   E_hp,   color=GREEN,  lw=1.5, label="HP fluid")
    ax.semilogy(t_nufi, E_nufi, color=PURPLE, lw=1.5, label="NuFI kinetic")

    if np.isfinite(gamma_hp) and np.isfinite(A_hp):
        t_fit = np.linspace(hp_window[0], hp_window[1], 300)
        ax.semilogy(t_fit, A_hp * np.exp(gamma_hp * t_fit),
                    ls="--", color=GREEN, lw=2,
                    label=f"HP fit  γ = {gamma_hp:.4f}  (R² = {r2_hp:.4f})")

    if np.isfinite(gamma_nufi) and np.isfinite(A_nufi):
        t_fit = np.linspace(nufi_window[0], nufi_window[1], 300)
        ax.semilogy(t_fit, A_nufi * np.exp(gamma_nufi * t_fit),
                    ls="--", color=PURPLE, lw=2,
                    label=f"NuFI fit  γ = {gamma_nufi:.4f}  (R² = {r2_nufi:.4f})")

    ax.set_xlabel(r"$t$", fontsize=13)
    ax.set_ylabel(r"$|E_k|$", fontsize=13)
    ax.set_title(
        r"Two-stream instability: linear phase growth rate ($k=0.5$, $v_0/v_\mathrm{th}=2.5$)",
        fontsize=13,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=11)
    ax.grid(which="both", ls="--", lw=0.4)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "growth_rate_fit.png")
    plt.savefig(outfile, dpi=180)
    print(f"Saved {outfile}")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--hp-window",   nargs=2, type=float, required=True,
                        metavar=("T_START", "T_END"),
                        help="Fit window for HP fluid  [t_start  t_end]")
    parser.add_argument("--nufi-window", nargs=2, type=float, required=True,
                        metavar=("T_START", "T_END"),
                        help="Fit window for NuFI kinetic  [t_start  t_end]")
    parser.add_argument("--hp-file",
                        default=os.path.join(_DATA_DIR, "efield_data_twostream.npz"),
                        help="Path to HP .npz file")
    parser.add_argument("--nufi-file",
                        default=os.path.join(_DATA_DIR, "statistics.csv"),
                        help="Path to NuFI statistics.csv")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot, just print the gamma values")
    args = parser.parse_args()

    if not os.path.isfile(args.hp_file):
        sys.exit(f"HP file not found: {args.hp_file}")
    if not os.path.isfile(args.nufi_file):
        sys.exit(f"NuFI file not found: {args.nufi_file}")

    t_hp,   E_hp   = load_hp(args.hp_file)
    t_nufi, E_nufi = load_nufi(args.nufi_file)

    hp_window   = tuple(args.hp_window)
    nufi_window = tuple(args.nufi_window)

    gamma_hp,   A_hp,   r2_hp   = fit_window(t_hp,   E_hp,   *hp_window)
    gamma_nufi, A_nufi, r2_nufi = fit_window(t_nufi, E_nufi, *nufi_window)

    print()
    print("=" * 52)
    print(f"  HP   fit  window [{hp_window[0]:6.1f}, {hp_window[1]:6.1f}]")
    print(f"    γ = {gamma_hp:.5f}   R² = {r2_hp:.5f}")
    print()
    print(f"  NuFI fit  window [{nufi_window[0]:6.1f}, {nufi_window[1]:6.1f}]")
    print(f"    γ = {gamma_nufi:.5f}   R² = {r2_nufi:.5f}")
    print()
    print("=" * 52)
    print()

    if not args.no_plot:
        make_plot(t_hp, E_hp, t_nufi, E_nufi,
                  hp_window, nufi_window,
                  gamma_hp, A_hp, r2_hp,
                  gamma_nufi, A_nufi, r2_nufi,
                  outdir=_DATA_DIR)


if __name__ == "__main__":
    main()
