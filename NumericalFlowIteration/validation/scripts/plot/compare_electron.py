#!/usr/bin/env python3
"""
Compare NuFI GPU + Hammett-Perkins fluid electron Landau damping sweeps.

Reads (per model subdir):
    validation/results/{model}/nufi_electron_k_results.txt  (k  gamma) — NuFI kinetic
    validation/results/{model}/hp_electron_k_results.txt    (k  gamma) — HP fluid
Writes:
    validation/results/{model}/nufi_electron_k_sweep.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


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
    parser.add_argument("--models", nargs="+", default=None, metavar="MODEL",
                        help="One or more model names to overlay "
                             "(loads results/{model}/hp_electron_k_results.txt). "
                             "If omitted, all subdirs of results/ are loaded.")
    parser.add_argument("--nufi", default=None,
                        help="Explicit path to NuFI kinetic results file.")
    args = parser.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(script_dir, "..", "..", "results")

    if args.models:
        model_dirs = args.models
    elif os.path.isdir(results_root):
        model_dirs = sorted(os.listdir(results_root))
    else:
        model_dirs = []

    hp_entries = []
    for model in model_dirs:
        path = os.path.join(results_root, model, "hp_electron_k_results.txt")
        data = load_results(path)
        if data is not None:
            hp_entries.append((model, data))
        else:
            print(f"Warning: no HP results found for model '{model}' ({path})")

    if args.nufi:
        nufi_data = load_results(args.nufi)
    else:
        nufi_data = None
        for model in model_dirs:
            path = os.path.join(results_root, model, "nufi_electron_k_results.txt")
            nufi_data = load_results(path)
            if nufi_data is not None:
                break

    if nufi_data is None and not hp_entries:
        print("No results files found.")
        sys.exit(1)

    # Okabe-Ito palette — safe for deuteranopia, protanopia, tritanopia
    OI_MODELS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442"]
    OI_NUFI   = "#0072B2"  # blue — distinct from all model colors

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, data) in enumerate(hp_entries):
        c = OI_MODELS[i % len(OI_MODELS)]
        k_hp, g_hp = data[:, 0], data[:, 1]
        ax.plot(k_hp,    -g_hp, ls="--", lw=1.5, color=c, zorder=2 + 2*i)
        ax.scatter(k_hp, -g_hp, s=60, color=c, label=label, zorder=3 + 2*i)

    if nufi_data is not None:
        k_nufi  = nufi_data[:, 0]
        g_nufi  = nufi_data[:, 1]
        ax.plot(k_nufi,    -g_nufi, ls="--", lw=1.5, color=OI_NUFI, zorder=10)
        ax.scatter(k_nufi, -g_nufi, s=60, color=OI_NUFI,
                   label="NuFI (kinetic GPU)", zorder=11)

    ax.set_xlabel(r"$k\lambda_D$", fontsize=13)
    ax.set_ylabel(r"$-\gamma / \omega_{pe}$ (damping rate)", fontsize=13)
    ax.set_title(r"Electron Landau damping rate vs $k$", fontsize=13)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=11)
    ax.grid(which="both", ls="--", lw=0.5)

    plt.tight_layout()
    if model_dirs:
        save_dir = os.path.join(results_root, model_dirs[0])
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = results_root
    outfile = os.path.join(save_dir, "nufi_electron_k_sweep.png")
    plt.savefig(outfile, dpi=200)
    print(f"Saved {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
