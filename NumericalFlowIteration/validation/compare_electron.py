#!/usr/bin/env python3
"""
Compare NuFI GPU + Hammett-Perkins fluid electron Landau damping sweeps
against the analytical Landau dispersion relation.

Reads:
    validation/nufi_electron_k_results.txt    (k  gamma)  — NuFI kinetic
    validation/nufi_hp_electron_k_results.txt (k  gamma)  — HP fluid (optional)
Writes:
    validation/nufi_electron_k_sweep.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from plasmadisp import electrostatic
    HAS_PLASMADISP = True
except ImportError:
    HAS_PLASMADISP = False
    print("Warning: plasmadisp not available — analytical curve will be skipped.")


def load_results(path):
    if not os.path.isfile(path):
        return None
    data = np.loadtxt(path)
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    nufi_file   = os.path.join(script_dir, "nufi_electron_k_results.txt")
    hp_file     = os.path.join(script_dir, "nufi_hp_electron_k_results.txt")

    nufi_data = load_results(nufi_file)
    hp_data   = load_results(hp_file)

    if nufi_data is None and hp_data is None:
        print("No results files found.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Collect all k values to set analytic range
    all_k = []
    if nufi_data is not None:
        all_k.extend(nufi_data[:, 0].tolist())
    if hp_data is not None:
        all_k.extend(hp_data[:, 0].tolist())
    k_lo, k_hi = min(all_k), max(all_k)

    # --- Analytical Landau curve ---
    if HAS_PLASMADISP:
        k_analytic     = np.linspace(k_lo, k_hi, 300)
        gamma_analytic = np.zeros(300)
        for i, k in enumerate(k_analytic):
            root = electrostatic.get_roots_to_electrostatic_dispersion(
                wp_e=1.0, vth_e=1.0, k0=k
            )
            gamma_analytic[i] = root.imag
        ax.plot(k_analytic, -gamma_analytic, ls="-", lw=2, color="blue",
                label="Analytical (Landau)", zorder=1)

    # --- Hammett-Perkins fluid ---
    if hp_data is not None:
        k_hp    = hp_data[:, 0]
        g_hp    = hp_data[:, 1]
        ax.plot(k_hp,    -g_hp, ls="--", lw=1.5, color="orange", zorder=2)
        ax.scatter(k_hp, -g_hp, s=60,    color="orange",
                   label="Hammett-Perkins (fluid)", zorder=3)

    # --- NuFI kinetic ---
    if nufi_data is not None:
        k_nufi  = nufi_data[:, 0]
        g_nufi  = nufi_data[:, 1]
        ax.plot(k_nufi,    -g_nufi, ls="--", lw=1.5, color="red", zorder=4)
        ax.scatter(k_nufi, -g_nufi, s=60,    color="red",
                   label="NuFI (kinetic GPU)", zorder=5)

    ax.set_xlabel(r"$k\lambda_D$")
    ax.set_ylabel(r"$-\gamma / \omega_{pe}$ (damping rate)")
    ax.set_title("Electron Landau damping rate vs $k$")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(which="both", ls="--", lw=0.5)

    plt.tight_layout()
    outfile = os.path.join(script_dir, "nufi_electron_k_sweep.png")
    plt.savefig(outfile, dpi=200)
    print(f"Saved {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
