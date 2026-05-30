#!/usr/bin/env python3
"""
Compare NuFI GPU + Hammett-Perkins fluid ion acoustic tau sweep
against the analytical Landau dispersion relation.

Reads:
    validation/nufi_ion_tau_results.txt       (tau  gamma) — NuFI kinetic
    validation/nufi_hp_ion_tau_results.txt    (tau  gamma) — HP fluid (optional)
Writes:
    validation/nufi_ion_tau_k{k:.4f}.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def find_physical_root(tau, k, cs=1.0):
    """Fluid ion-acoustic dispersion (Chapurin Eq. 2.49)."""
    alpha = np.sqrt(8.0 / np.pi)
    a2k2  = alpha**2 * k**2 * cs**2
    k2cs2 = k**2 * cs**2

    c4 =  1.0
    c3 =  0.0
    c2 =  tau * a2k2 - k2cs2 * (1.0 + tau) - 2.0 * tau * k2cs2
    c1 =  1j * tau**1.5 * alpha**3 * k**3 * cs**3
    c0 = -k2cs2 * (1.0 + tau) * tau * a2k2

    roots  = np.roots([c4, c3, c2, c1, c0])
    damped = roots[roots.imag < 0]
    if len(damped) == 0:
        return None
    return damped[np.argmin(np.abs(damped.imag / (np.abs(damped.real) + 1e-30)))]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, required=True,
                        help="Wavenumber used in the sweep")
    parser.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                        help="One or more model names to overlay "
                             "(loads results/{model}/hp_ion_tau_results.txt). "
                             "If omitted, all subdirs of results/ are loaded.")
    parser.add_argument("--nufi", default=None,
                        help="Explicit path to NuFI kinetic results file. "
                             "Default: results/{first_model}/nufi_ion_tau_results.txt")
    args = parser.parse_args()
    k = args.k

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(script_dir, "..", "..", "results")

    # Determine which model directories to load
    if args.models:
        model_dirs = args.models
    elif os.path.isdir(results_root):
        model_dirs = sorted(os.listdir(results_root))
    else:
        model_dirs = []

    # Load HP results for each requested model
    hp_entries = []
    for model in model_dirs:
        path = os.path.join(results_root, model, "hp_ion_tau_results.txt")
        data = load_results(path)
        if data is not None:
            hp_entries.append((model, data))
        else:
            print(f"Warning: no HP results found for model '{model}' ({path})")

    # NuFI results: explicit path, or first model dir that has the file
    if args.nufi:
        nufi_data = load_results(args.nufi)
    else:
        nufi_data = None
        for model in model_dirs:
            path = os.path.join(results_root, model, "nufi_ion_tau_results.txt")
            nufi_data = load_results(path)
            if nufi_data is not None:
                break

    if nufi_data is None and not hp_entries:
        print("No results files found.")
        sys.exit(1)

    # Collect tau range for analytical curve
    all_tau = []
    if nufi_data is not None:
        all_tau.extend(nufi_data[:, 0].tolist())
    for _, d in hp_entries:
        all_tau.extend(d[:, 0].tolist())
    tau_lo, tau_hi = min(all_tau), max(all_tau)

    # Analytical ion-acoustic dispersion
    tau_analytic   = np.linspace(tau_lo, tau_hi, 300)
    gamma_analytic = np.zeros(300)
    for i, tau in enumerate(tau_analytic):
        root = find_physical_root(tau, k)
        if root is not None:
            gamma_analytic[i] = -root.imag

    fig, ax = plt.subplots(figsize=(8, 5))

    # Okabe-Ito palette — safe for deuteranopia, protanopia, tritanopia
    OI_ANALYTIC = "#000000"  # black — universally distinguishable
    OI_MODELS   = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442"]
    OI_NUFI     = "#0072B2"  # blue — matches electron plot NuFI color

    ax.plot(tau_analytic, gamma_analytic, ls="-", lw=2, color=OI_ANALYTIC,
            label="Analytical (Linearized 3HP)", zorder=1)

    for i, (label, data) in enumerate(hp_entries):
        c = OI_MODELS[i % len(OI_MODELS)]
        tau_hp, g_hp = data[:, 0], data[:, 1]
        ax.plot(tau_hp,    -g_hp, ls="--", lw=1.5, color=c, zorder=2 + 2*i)
        ax.scatter(tau_hp, -g_hp, s=60, color=c, label=label, zorder=3 + 2*i)

    if nufi_data is not None:
        tau_nufi = nufi_data[:, 0]
        g_nufi   = nufi_data[:, 1]
        ax.plot(tau_nufi,    -g_nufi, ls="--", lw=1.5, color=OI_NUFI, zorder=10)
        ax.scatter(tau_nufi, -g_nufi, s=60, color=OI_NUFI,
                   label="NuFI (kinetic GPU)", zorder=11)

    ax.set_xlabel(r"$\tau = T_{0i}/T_{0e}$", fontsize=13)
    ax.set_ylabel(r"$-\gamma$ (damping rate)", fontsize=13)
    ax.set_title(r"Ion acoustic Landau damping vs $\tau$   ($L = 100$)", fontsize=13)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=11)
    ax.grid(which="both", ls="--", lw=0.5)

    plt.tight_layout()
    # Save into the first model subdir that has data (or results_root if none)
    if model_dirs:
        save_dir = os.path.join(results_root, model_dirs[0])
    else:
        save_dir = results_root
    outfile = os.path.join(save_dir, f"nufi_ion_tau_k{k:.4f}.png")
    plt.savefig(outfile, dpi=200)
    print(f"Saved {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
